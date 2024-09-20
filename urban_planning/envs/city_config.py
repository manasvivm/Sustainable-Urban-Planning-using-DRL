NON_BLOCK_LAND_USE = (
    'outside',
    'feasible',
    'road',
    'traffic_line',
    'boundary'
)

BLOCK_LAND_USE = (
    'residential',
    'business',
    'wastemgmt',
    'green_l',
    'green_s',
    'school',
    'hospital_l',
    'hospital_s',
    'recreation',
    'office'
)

LAND_USE = (
    NON_BLOCK_LAND_USE + BLOCK_LAND_USE)

OUTSIDE = 0
FEASIBLE = 1
ROAD = 2
TRAFFIC_LINE =3
BOUNDARY = 4
RESIDENTIAL = 5
BUSINESS = 6
WASTEMGMT = 7
GREEN_L = 8
GREEN_S = 9
SCHOOL = 10
HOSPITAL_L = 11
HOSPITAL_S = 12
RECREATION = 13
OFFICE = 14

LAND_USE_ID = (
    OUTSIDE,
    FEASIBLE,
    ROAD,
    TRAFFIC_LINE,
    BOUNDARY,
    RESIDENTIAL,
    BUSINESS,
    WASTEMGMT,
    GREEN_L,
    GREEN_S,
    SCHOOL,
    HOSPITAL_L,
    HOSPITAL_S,
    RECREATION,
    OFFICE
)

NUM_TYPES = len(LAND_USE_ID)

LAND_USE_ID_MAP = dict(
    zip(LAND_USE, LAND_USE_ID))

LAND_USE_ID_MAP_INV = dict(
    zip(LAND_USE_ID, LAND_USE))

INTERSECTION = 15

PUBLIC_SERVICES_ID = (
    BUSINESS,
    WASTEMGMT,
    SCHOOL,
    (HOSPITAL_L, HOSPITAL_S),
    RECREATION,
    OFFICE
)

PUBLIC_SERVICES = (
    'shopping',
    'wastemgmt',
    'education',
    'medical care',
    'entertainment',
    'working'
)

GREEN_ID = (
    GREEN_L,
    GREEN_S
)
WASTEMGMT_ID = WASTEMGMT
GREEN_AREA_THRESHOLD = 1500
WASTEMGMT_AREA_THRESHOLD = 500

TYPE_COLOR_MAP = {
    'boundary': 'lightgreen',
    'business': 'fuchsia',
    'feasible': 'white',
    'green_l': 'green',
    'green_s': 'lightgreen',
    'hospital_l': 'blue',
    'hospital_s': 'cyan',
    'wastemgmt': 'gold',
    'outside': 'black',
    'residential': 'yellow',
    'road': 'red',
    'traffic_line': 'green',
    'school': 'darkorange',
    'recreation': 'lavender',
    'office' : 'red'
}
