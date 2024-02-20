import psycopg2

conn = psycopg2.connect(database="data_heterog",
                        host="10.0.0.99",
                        user="postgres",
                        password="sX2023!?!",
                        port="5432")

# For Power BI
#Driver={PostgreSQL ANSI(x64)}; Server=10.0.0.99; Port=5432; Database= data_heterog