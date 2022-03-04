import matplotlib.pyplot as plt
import numpy as np
from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper, JSON, N3
from pprint import pprint
import pickle


def acceptable_subject(name):
    if ":" in name:
        return False
    return True


def query(entity, entities_label, label):
    escapes = ['(', ')', "'"]
    for escape in escapes:
        entity = entity.replace(escape, '\\' + escape)

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
    filtered_results = [result for result in results if acceptable_subject(result)]

    for result in filtered_results:
        if label == 3:
            if result not in entities_label:
                entities_label[result] = label
        else:
            entities_label[result] = label

    print(f"Finish querying {entity}, total valid dbr {len(filtered_results)}")

    return filtered_results


def BFS_layered(queue, edges, entities_label, label):
    entities_label[queue[0]] = label
    layer = 2
    for l in range(layer):
        label = 3 if l == 1 else label
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

    entity_list = list(entities)
    entity_hashmap = {entity: i for i, entity in enumerate(entity_list)}

    print(f"len(edges) = {len(edges)}")
    edges = [edge if entity_hashmap[edge[0]] < entity_hashmap[edge[1]] else edge[::-1] for edge in edges]
    edges = list(set(edges))
    print(f"len(edges) = {len(edges)}")
    print(edges[:10])

    edge_inds = np.array([[entity_hashmap[edge[0]], entity_hashmap[edge[1]]] for edge in edges])
    labels = np.array([entities_label[entity] for entity in entity_list])

    # print(labels)
    for i in range(np.max(labels) + 1):
        print(f"labels contain {np.sum(labels == i)} {i}")

    np.save(f"data/numpy/edge_inds.npy", edge_inds)
    np.save(f"data/numpy/node_labels.npy", labels)

    with open('data/pickle/entity_names.pickle', 'wb') as handle:
        pickle.dump(entity_list, handle)

    with open('data/pickle/entity_hashmap.pickle', 'wb') as handle:
        pickle.dump(entity_hashmap, handle)


# def exp():
#     results = query('List_of_manufacturing_processes')
#     print(len(results))
#     for i in range(len(results)):
#         r = query(results[i])
#         print(len(r))
#         # print(r[:3])
#         print(f"step {i}")
 

if __name__ == "__main__":
    BFS()
 
