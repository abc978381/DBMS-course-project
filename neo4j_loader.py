from neo4j import GraphDatabase
import re

class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def setup_schema(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")

    def format_relation(self, relation_text):
        clean = re.sub(r'[^\w\s]', '', relation_text)
        return re.sub(r'\s+', '_', clean).upper()

    def load_triples(self, triples):
        count = 0
        with self.driver.session() as session:
            for t in triples:
                rel_type = self.format_relation(t['relation'])
                if not rel_type: 
                    continue    
                query = f"""
                MERGE (s:Entity {{name: toLower($subject)}})
                MERGE (o:Entity {{name: toLower($obj)}})
                MERGE (s)-[r:`{rel_type}`]->(o)
                ON CREATE SET r.sentence = $sentence, r.confidence = $confidence, r.source = $source
                ON MATCH SET r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END
                """
                session.run(query, 
                            subject=t['subject'], 
                            obj=t['obj'], 
                            sentence=t['sentence'], 
                            confidence=t.get('confidence', 1.0),
                            source=t.get('source', 'unknown'))
                count += 1
        print(f"Successfully merged {count} relationships into Neo4j.")