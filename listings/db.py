import pg8000.dbapi

from listings.config import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER

THUMBNAIL_POSITION = 0
SECONDARY_POSITION = 1
REAL_GENDERS = ("Men", "Women")

LISTINGS_WITH_THUMBNAIL_AND_SECONDARY_PHOTOS = f"""
SELECT general_listings.id,
       general_listings.brand,
       general_listings.category,
       general_listings.gender,
       general_listings.color,
       general_listings.description,
       (SELECT general_images.s3_url
          FROM general_images
         WHERE general_images.listing_id = general_listings.id
           AND general_images.position = {THUMBNAIL_POSITION}) AS thumbnail_url,
       (SELECT general_images.s3_url
          FROM general_images
         WHERE general_images.listing_id = general_listings.id
           AND general_images.position = {SECONDARY_POSITION}) AS secondary_url
  FROM general_listings
 WHERE general_listings.description LIKE '💎%'
   AND general_listings.gender IN {REAL_GENDERS}
 ORDER BY general_listings.id
"""


def pg_connect():
    return pg8000.dbapi.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
    )


def execute_query_command(sql: str, params=None) -> list[dict]:
    conn = pg_connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        column_names = [column[0] for column in cur.description]
        return [dict(zip(column_names, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def execute_write_commands(sql: str, params_per_row: list[tuple]) -> int:
    conn = pg_connect()
    try:
        cur = conn.cursor()
        for params in params_per_row:
            cur.execute(sql, params)
        conn.commit()
        return len(params_per_row)
    finally:
        conn.close()


def has_both_photos(row: dict) -> bool:
    return row["thumbnail_url"] is not None and row["secondary_url"] is not None


def fetch_listings_with_photos() -> list[dict]:
    rows = execute_query_command(LISTINGS_WITH_THUMBNAIL_AND_SECONDARY_PHOTOS)
    return [row for row in rows if has_both_photos(row)]
