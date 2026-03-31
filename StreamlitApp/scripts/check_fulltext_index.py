"""
Check if Neo4j fulltext index exists and test it.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
import config

driver = GraphDatabase.driver(
    config.NEO4J_URI,
    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

with driver.session(database=config.NEO4J_DATABASE) as session:
    # Check if index exists
    print("=== Checking fulltext indexes ===")
    result = session.run("SHOW FULLTEXT INDEXES")
    indexes = list(result)
    
    if indexes:
        print(f"Found {len(indexes)} fulltext indexes:")
        for idx in indexes:
            print(f"  - {idx}")
    else:
        print("❌ NO FULLTEXT INDEXES FOUND!")
    
    print("\n=== Checking Chunk nodes ===")
    result = session.run("MATCH (c:Chunk) RETURN count(c) as count, c.text IS NOT NULL as has_text LIMIT 1")
    for record in result:
        print(f"Chunk nodes: {record['count']}, has text: {record['has_text']}")
    
    print("\n=== Testing direct fulltext query ===")
    try:
        result = session.run("""
            CALL db.index.fulltext.queryNodes('chunk_fulltext', 'rice water')
            YIELD node, score
            RETURN count(node) as count
        """)
        for record in result:
            print(f"Direct query result: {record['count']} nodes")
    except Exception as e:
        print(f"❌ Error querying fulltext index: {e}")

driver.close()
