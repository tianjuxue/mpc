import matplotlib.pyplot as plt
import numpy as np
from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper, JSON, N3
from pprint import pprint


def query():

    sparql = SPARQLWrapper('https://dbpedia.org/sparql')
    sparql.setQuery('''
        SELECT ?object
        WHERE { dbr:Barack_Obama rdfs:label ?object .}
        # WHERE { dbr:Barack_Obama dbo:abstract ?object .}
    ''')
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert()

