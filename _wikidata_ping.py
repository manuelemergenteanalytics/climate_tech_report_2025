import requests
q = "SELECT * WHERE { ?s ?p ?o } LIMIT 1"
r = requests.get("https://query.wikidata.org/sparql",
                 params={"format": "json", "query": q},
                 headers={"User-Agent": "ctr25/test"})
print(r.status_code, r.ok)
