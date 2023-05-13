import numpy as np
import kgbench.load as kgload
#from kgbench.load import fastload, getfile
URI_PREFIX = "http://multimodal-knowledge-graph-augmentation.com/"

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

# class Data:
#     """
#     Class representing a dataset. 
#     Being an alternative version of the Data class in kgbench, adapted for the evaluation framework.
#     Explicitly defining values not None, using the final torch distribution and setting the dataset name.

#     """
    

#     def __init__(self, data:kgload.Data):

#         self.name:str = data.name
#         self.torch = True
#         self.triples = fastload(getfile(dir, 'triples.int.csv.gz'))
#         self.i2r, self.r2i = load_indices(getfile(dir, 'relations.int.csv'))
#         self.i2e, self.e2i = load_entities(getfile(dir, 'nodes.int.csv'))

#         self.num_entities  = len(self.i2e)
#         self.num_relations = len(self.i2r)

#         train, val, test = \
#             np.loadtxt(getfile(dir, 'training.int.csv'),   dtype=np.int64, delimiter=',', skiprows=1), \
#             np.loadtxt(getfile(dir, 'validation.int.csv'), dtype=np.int64, delimiter=',', skiprows=1), \
#             np.loadtxt(getfile(dir, 'testing.int.csv'),    dtype=np.int64, delimiter=',', skiprows=1)

#             if final and catval:
#                 self.training = np.concatenate([train, val], axis=0)
#                 self.withheld = test

#                 if name not in ['aifb', 'mutag', 'bgs', 'am']:
#                     warnings.warn('Adding the validation set to the training data. Note that this is not the correct '
#                                   'way to load the KGBench data, and will lead to inflated performance. For AIFB, '
#                                   'MUTAG, BGS and AM, this is the correct way to load the data.')
#             elif final:
#                 if name in ['aifb', 'mutag', 'bgs', 'am']:
#                     warnings.warn('The validation data is not added to the training data. For AIFB, MUTAG, BGS and AM, '
#                                   'the correct evaluation is to combine train and validation for the final evaluation run.'
#                                   'Set include_val to True when loading the data.')

#                 self.training = train
#                 self.withheld = test
#             else:
#                 self.training = train
#                 self.withheld = val

#             self.final = final

#             self.num_classes = len(set(self.training[:, 1]))

#             # print(f'   {len(self.triples)} triples')

#             if use_torch: # this should be constant-time/memory
#                 self.triples = torch.from_numpy(self.triples)
#                 self.training = torch.from_numpy(self.training)
#                 self.withheld = torch.from_numpy(self.withheld)

#         self.triples = None
#         """ The edges of the knowledge graph (the triples), represented by their integer indices. A (m, 3) numpy 
#             or pytorch array.
#         """

#         self.i2r, self.r2i = None, None

#         self.i2e = None
#         """ A mapping from an integer index to an entity representation. An entity is either a simple string indicating the label 
#             of the entity (a url, blank node or literal), or it is a pair indicating the datatype and the label (in that order).
#         """

#         self.e2i = None
#         """ A dictionary providing the inverse mappring of i2e
#         """

#         self.num_entities = None
#         """ Total number of distinct entities (nodes) in the graph """

#         self.num_relations = None
#         """ Total number of distinct relation types in the graph """

#         self.num_classes = None
#         """ Total number of classes in the classification task """

#         self.training = None
#         """ Training data: a matrix with entity indices in column 0 and class indices in column 1.
#             In non-final mode, this is the training part of the train/val/test split. In final mode, the training part, 
#             possibly concatenated with the validation data.
#         """

#         self.withheld = None
#         """ Validation/testing data: a matrix with entity indices in column 0 and class indices in column 1.
#             In non-final mode this is the validation data. In final mode this is the testing data.
#         """

#         self._dt_l2g = {}
#         self._dt_g2l = {}

#         self._datatypes = None
#         if dir is not None:

#             self.torch = use_torch

#             self.triples = fastload(getfile(dir, 'triples.int.csv.gz'))

#             self.i2r, self.r2i = load_indices(getfile(dir, 'relations.int.csv'))
#             self.i2e, self.e2i = load_entities(getfile(dir, 'nodes.int.csv'))

#             self.num_entities  = len(self.i2e)
#             self.num_relations = len(self.i2r)

#             train, val, test = \
#                 np.loadtxt(getfile(dir, 'training.int.csv'),   dtype=np.int64, delimiter=',', skiprows=1), \
#                 np.loadtxt(getfile(dir, 'validation.int.csv'), dtype=np.int64, delimiter=',', skiprows=1), \
#                 np.loadtxt(getfile(dir, 'testing.int.csv'),    dtype=np.int64, delimiter=',', skiprows=1)

#             if final and catval:
#                 self.training = np.concatenate([train, val], axis=0)
#                 self.withheld = test

#                 if name not in ['aifb', 'mutag', 'bgs', 'am']:
#                     warnings.warn('Adding the validation set to the training data. Note that this is not the correct '
#                                   'way to load the KGBench data, and will lead to inflated performance. For AIFB, '
#                                   'MUTAG, BGS and AM, this is the correct way to load the data.')
#             elif final:
#                 if name in ['aifb', 'mutag', 'bgs', 'am']:
#                     warnings.warn('The validation data is not added to the training data. For AIFB, MUTAG, BGS and AM, '
#                                   'the correct evaluation is to combine train and validation for the final evaluation run.'
#                                   'Set include_val to True when loading the data.')

#                 self.training = train
#                 self.withheld = test
#             else:
#                 self.training = train
#                 self.withheld = val

#             self.final = final

#             self.num_classes = len(set(self.training[:, 1]))

#             # print(f'   {len(self.triples)} triples')

#             if use_torch: # this should be constant-time/memory
#                 self.triples = torch.from_numpy(self.triples)
#                 self.training = torch.from_numpy(self.training)
#                 self.withheld = torch.from_numpy(self.withheld)