import pytest

from listings import db
from listings.db import execute_query_command, execute_write_commands, fetch_listings_with_photos, has_both_photos

INSERT_BRAND = "INSERT INTO fake_brands (id, name) VALUES (%s, %s)"


class FakeCursor:
    def __init__(self, column_names, rows, failing_sql=None):
        self.description = [(column_name,) for column_name in column_names]
        self.rows = rows
        self.failing_sql = failing_sql
        self.executed = []

    def execute(self, sql, params):
        if sql == self.failing_sql:
            raise RuntimeError("fake database error")
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, column_names=(), rows=(), failing_sql=None):
        self.fake_cursor = FakeCursor(column_names, rows, failing_sql)
        self.commit_count = 0
        self.closed = False

    def cursor(self):
        return self.fake_cursor

    def commit(self):
        self.commit_count += 1

    def close(self):
        self.closed = True


def connect_to(monkeypatch, connection: FakeConnection) -> FakeConnection:
    monkeypatch.setattr(db, "pg_connect", lambda: connection)
    return connection


def test_pg_connect_opens_a_connection_with_the_postgres_settings(monkeypatch):
    connect_arguments = {}
    monkeypatch.setattr(db.pg8000.dbapi, "connect", lambda **arguments: connect_arguments.update(arguments))
    monkeypatch.setattr(db, "POSTGRES_USER", "fake_user")
    monkeypatch.setattr(db, "POSTGRES_PASSWORD", "fake_password")
    monkeypatch.setattr(db, "POSTGRES_HOST", "fake.host")
    monkeypatch.setattr(db, "POSTGRES_PORT", 15432)
    monkeypatch.setattr(db, "POSTGRES_DB", "fake_db")

    db.pg_connect()

    assert connect_arguments == {
        "user": "fake_user",
        "password": "fake_password",
        "host": "fake.host",
        "port": 15432,
        "database": "fake_db",
    }


def test_execute_query_command_returns_each_row_as_a_dict_keyed_by_column_name(monkeypatch):
    connect_to(monkeypatch, FakeConnection(column_names=["id", "brand"], rows=[(1, "Russell"), (2, "Levi's")]))
    assert execute_query_command("SELECT fake") == [{"id": 1, "brand": "Russell"}, {"id": 2, "brand": "Levi's"}]


def test_execute_query_command_passes_the_parameters_to_the_cursor(monkeypatch):
    connection = connect_to(monkeypatch, FakeConnection(column_names=["id"]))
    execute_query_command("SELECT fake WHERE id = %s", (7,))
    assert connection.fake_cursor.executed == [("SELECT fake WHERE id = %s", (7,))]


def test_execute_query_command_closes_the_connection(monkeypatch):
    connection = connect_to(monkeypatch, FakeConnection(column_names=["id"]))
    execute_query_command("SELECT fake")
    assert connection.closed


def test_execute_query_command_closes_the_connection_when_the_query_fails(monkeypatch):
    connection = connect_to(monkeypatch, FakeConnection(failing_sql="SELECT broken"))
    with pytest.raises(RuntimeError):
        execute_query_command("SELECT broken")
    assert connection.closed


def test_execute_write_commands_runs_the_statement_once_per_row_of_parameters(monkeypatch):
    connection = connect_to(monkeypatch, FakeConnection())
    execute_write_commands(INSERT_BRAND, [(1, "Russell"), (2, "Champion")])
    assert connection.fake_cursor.executed == [(INSERT_BRAND, (1, "Russell")), (INSERT_BRAND, (2, "Champion"))]


def test_execute_write_commands_commits_once_for_the_whole_batch(monkeypatch):
    connection = connect_to(monkeypatch, FakeConnection())
    execute_write_commands(INSERT_BRAND, [(1, "Russell"), (2, "Champion")])
    assert connection.commit_count == 1


def test_execute_write_commands_returns_the_number_of_rows_written(monkeypatch):
    connect_to(monkeypatch, FakeConnection())
    assert execute_write_commands(INSERT_BRAND, [(1, "Russell"), (2, "Champion")]) == 2


def test_execute_write_commands_closes_the_connection(monkeypatch):
    connection = connect_to(monkeypatch, FakeConnection())
    execute_write_commands(INSERT_BRAND, [(1, "Russell")])
    assert connection.closed


def test_execute_write_commands_commits_nothing_when_a_statement_fails(monkeypatch):
    connection = connect_to(monkeypatch, FakeConnection(failing_sql=INSERT_BRAND))
    with pytest.raises(RuntimeError):
        execute_write_commands(INSERT_BRAND, [(1, "Russell")])
    assert connection.commit_count == 0
    assert connection.closed


def test_has_both_photos_is_true_with_a_thumbnail_and_a_secondary_photo():
    assert has_both_photos({"thumbnail_url": "https://fake/7_a.jpg", "secondary_url": "https://fake/7_b.jpg"})


def test_has_both_photos_is_false_without_a_thumbnail():
    assert not has_both_photos({"thumbnail_url": None, "secondary_url": "https://fake/7_b.jpg"})


def test_has_both_photos_is_false_without_a_secondary_photo():
    assert not has_both_photos({"thumbnail_url": "https://fake/7_a.jpg", "secondary_url": None})


def test_fetch_listings_with_photos_drops_rows_missing_either_photo(monkeypatch):
    rows = [
        {"id": 1, "thumbnail_url": "https://fake/1_a.jpg", "secondary_url": "https://fake/1_b.jpg"},
        {"id": 2, "thumbnail_url": None, "secondary_url": "https://fake/2_b.jpg"},
        {"id": 3, "thumbnail_url": "https://fake/3_a.jpg", "secondary_url": None},
    ]
    monkeypatch.setattr(db, "execute_query_command", lambda sql: rows)
    assert [row["id"] for row in fetch_listings_with_photos()] == [1]
