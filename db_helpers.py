import db_conn


datasets = {"Skin Cancer" : 1,"MRNET" : 2, "Pneumnonia" : 3,  "ABIDE" : 4, "Cataracts": 5, "IRMA" : 6,}


sample_sizes = {
    "Skin Cancer" : [3297, 1648, 500, 250],
    "MRNET" : [1008,504,300,150],
    "Pneumnonia" : [2928, 1000, 500, 250],
    "ABIDE" : [452, 226, 100, 50],
    "Cataracts" : [2254, 1127, 500, 200],
    "IRMA" : []
}


# def populateDatasets(dataset_name, sample_size, num_control, num_treatment):
#     conn = db_conn.conn
#     cursor = conn.cursor()

#     sql_query = f"INSERT INTO datasets(dataset_name, sample_size, num_control, num_treatment) VALUES('{dataset_name}',{sample_size},{num_control},{num_treatment});"

#     cursor.execute(sql_query)

#     conn.commit()
#     cursor.close()
#     conn.close()



# def populateResults(sql_query):
#     conn = db_conn.conn
#     cursor = conn.cursor()

#     cursor.execute(sql_query)

#     conn.commit()
#     cursor.close()
#     conn.close()


def getNextInstanceID():
    """ This function returns the current instance ID for logging purposes."""
    import numpy as np
    conn = db_conn.conn
    cursor = conn.cursor()

    sql_query = "SELECT MAX(instance_id) FROM results;"

    cursor.execute(sql_query)
    instance_id = int(cursor.fetchall()[0][0] or 0)

    cursor.close()
    conn.close()

    return instance_id + 1


# def getSampleSizes(dataset_id):
#     import numpy as np
#     conn = db_conn.conn
#     cursor = conn.cursor()
#     full_sql_query = f""" SELECT size_n FROM sample_sizes WHERE dataset_id = {dataset_id} AND id = 1;"""
#     half_sql_query = f""" SELECT size_n FROM sample_sizes WHERE dataset_id = {dataset_id} AND id = 2;"""
#     med_sql_query = f""" SELECT size_n FROM sample_sizes WHERE dataset_id = {dataset_id} AND id = 3;"""
#     small_sql_query = f""" SELECT size_n FROM sample_sizes WHERE dataset_id = {dataset_id} AND id = 4;"""

#     cursor.execute(full_sql_query)
#     full_sample_size = int(cursor.fetchall()[0][0])

#     cursor.execute(half_sql_query)
#     half_sample_size = int(cursor.fetchall()[0][0])

#     cursor.execute(med_sql_query)
#     med_sample_size = int(cursor.fetchall()[0][0])

#     cursor.execute(small_sql_query)
#     small_sample_size = int(cursor.fetchall()[0][0])

#     cursor.close()
#     conn.close()
#     return full_sample_size - 1, half_sample_size - 1, med_sample_size - 1, small_sample_size - 1


