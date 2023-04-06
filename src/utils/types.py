URI_PREFIX = "http://master-thesis.com/"

ALL_TYPES = [
    '@es', '@fy', '@nl', '@nl-nl', '@pt', '@ru',
    'blank_node',
    'http://kgbench.info/dt#base64Image',
    'http://www.opengis.net/ont/geosparql#wktLiteral',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#langString',
    'http://www.w3.org/2001/XMLSchema#anyURI',
    'http://www.w3.org/2001/XMLSchema#boolean',
    'http://www.w3.org/2001/XMLSchema#date',
    'http://www.w3.org/2001/XMLSchema#dateTime',
    'http://www.w3.org/2001/XMLSchema#decimal',
    'http://www.w3.org/2001/XMLSchema#gYear',
    'http://www.w3.org/2001/XMLSchema#nonNegativeInteger',
    'http://www.w3.org/2001/XMLSchema#positiveInteger',
    'http://www.w3.org/2001/XMLSchema#string',
    'iri',
    'none']

RDF_ENTITY_TYPES = [
    'iri', 'none', 'blank_node']

RDF_NUMBER_TYPES = [
    'http://www.w3.org/2001/XMLSchema#decimal',
    'http://www.w3.org/2001/XMLSchema#gYear',
    'http://www.w3.org/2001/XMLSchema#nonNegativeInteger',
    'http://www.w3.org/2001/XMLSchema#positiveInteger']

RDF_DECIMAL_TYPES = [
    'http://www.w3.org/2001/XMLSchema#decimal'
]

RDF_DATE_TYPES = [
    'http://www.w3.org/2001/XMLSchema#date',
    'http://www.w3.org/2001/XMLSchema#dateTime']

IMAGE_TYPES = [
    'http://kgbench.info/dt#base64Image'
]

GEO_TYPES = [
    'http://www.opengis.net/ont/geosparql#wktLiteral'
]

NONE_TYPES =[ 
    'none'
]

ALL_LITERALS = [
    '@es', '@fy', '@nl', '@nl-nl', '@pt', '@ru',
    'http://kgbench.info/dt#base64Image',
    'http://www.opengis.net/ont/geosparql#wktLiteral',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#langString',
    'http://www.w3.org/2001/XMLSchema#anyURI',
    'http://www.w3.org/2001/XMLSchema#boolean',
    'http://www.w3.org/2001/XMLSchema#date',
    'http://www.w3.org/2001/XMLSchema#dateTime',
    'http://www.w3.org/2001/XMLSchema#decimal',
    'http://www.w3.org/2001/XMLSchema#gYear',
    'http://www.w3.org/2001/XMLSchema#nonNegativeInteger',
    'http://www.w3.org/2001/XMLSchema#positiveInteger',
    'http://www.w3.org/2001/XMLSchema#string']

ALL_BUT_NUMBER = [
    '@es', '@fy', '@nl', '@nl-nl', '@pt', '@ru',
    'http://kgbench.info/dt#base64Image',
    'http://www.opengis.net/ont/geosparql#wktLiteral',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#langString',
    'http://www.w3.org/2001/XMLSchema#anyURI',
    'http://www.w3.org/2001/XMLSchema#boolean',
    'http://www.w3.org/2001/XMLSchema#date',
    'http://www.w3.org/2001/XMLSchema#dateTime',
    'http://www.w3.org/2001/XMLSchema#string']

POTENTIAL_TEXT_TYPES = [
    '@es', '@fy', '@nl', '@nl-nl', '@pt', '@ru',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#langString',
    'http://www.w3.org/2001/XMLSchema#string',
    'none'
]