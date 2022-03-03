import matplotlib.pyplot as plt
import numpy as np
from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper, JSON, N3
from pprint import pprint


def query(entity, entities_label, label):
    entities_label[entity] = label
    escapes = ['(', ')', "'"]
    for escape in escapes:
        entity = entity.replace(escape, '\\' + escape)

    # entity = entity.replace('(', '\(')
    # entity = entity.replace(')', '\)')
    sparql = SPARQLWrapper('https://dbpedia.org/sparql')
    try:
        sparql.setQuery(f'''SELECT ?object WHERE {{dbr:{entity} dbo:wikiPageWikiLink ?object .}}''')
        sparql.setReturnFormat(JSON)
        qres = sparql.query().convert()
    except Exception as e:
        print(f"\n\nraw error messenge: \n{e}")
        print(f"\n\nentity {entity} query fail or conversion fail\n")
        return []

    prefix = 'http://dbpedia.org/resource/'
    results = list(map(lambda x: x['object']['value'][len(prefix):], qres['results']['bindings']))
    print(f"querying {entity}, total dbr {len(results)}")
    for result in results:
        entities_label[result] = label

    return results


def BFS_layered(queue, edges, entities_label, label):
    layer = 2
    for _ in range(layer):
        new_queue = []
        for i, crt_entity in enumerate(queue):
            results = query(crt_entity, entities_label, label)
            edges += list(map(lambda x: (crt_entity, x), results))
            new_queue += results
            # if i > 3:
            #     break
        queue = new_queue


def BFS():
    entities_label = {}
    edges = []
    BFS_layered(['List_of_manufacturing_processes'], edges, entities_label, 0)
    BFS_layered(['Material'], edges, entities_label, 1)
    BFS_layered(['Energy'], edges, entities_label, 2)

    entities = set(sum(edges, ()))
    print(f"len(entities) = {len(entities)}")
    print(f"len(entities_label) = {len(entities_label)}")

    a = entities
    b = set(entities_label.keys())

    assert len(entities) == len(entities_label), f"len(entities) and len(entities_label) not equal!"

    entities_list = list(entities)
    entities_hashmap = {entity: i for i, entity in enumerate(entities_list)}

    print(f"len(edges) = {len(edges)}")
    edges = [edge if entities_hashmap[edge[0]] < entities_hashmap[edge[1]] else edge[::-1] for edge in edges]
    edges = list(set(edges))
    print(f"len(edges) = {len(edges)}")
    print(edges[:10])

    edge_inds = [[entities_hashmap[edge[0]], entities_hashmap[edge[1]]] for edge in edges]
    labels = [entities_label[entity] for entity in entities_list]

    np.save(f"data/numpy/edge_inds.npy", np.array(edge_inds))
    np.save(f"data/numpy/node_labels.npy", np.array(labels))
 

def exp():
    results = query('List_of_manufacturing_processes')
    print(len(results))
    for i in range(len(results)):
        r = query(results[i])
        print(len(r))
        # print(r[:3])
        print(f"step {i}")
 

if __name__ == "__main__":
    # query_batch(["List_of_manufacturing_processes", "Turning"])
    BFS()
 
