# -*- coding: utf-8 -*-
# codes used to interact with database

import pymysql
from contextlib import closing
import traceback
import json

class SqlOperator(object):

    def __init__(self, logger, conf):

        self.logger = logger

        self.db_conf = conf['DataBase']
        self.host = self.db_conf['host']
        self.port = int(self.db_conf['port'])
        self.user = self.db_conf['user']
        self.password = self.db_conf['password']
        self.database = self.db_conf['database']
        self.localhost = self.db_conf['localhost']

    def get_conn(self):

        self.connection = pymysql.connect(host=self.host,
                        user=self.user, passwd=self.password, db=self.database,
                        port=self.port, charset = 'utf8')
        return self.connection

    def get_from_sql(self, table, key, key_value):
        with closing(self.get_conn()) as conn:
            with closing(conn.cursor(pymysql.cursors.DictCursor)) as cur:
                cur.execute("select * from %s where %s = %s" % (table, key, key_value))
                return cur.fetchall()

    def update_to_sql(self, table, key, key_value, **kwargs):
        # Usage: update_to_sql(table = 'dataset' ,key = 'featureStatus',key_value = 10,**{'patientCount':10, 'nodesCount':20})
        with closing(self.get_conn()) as conn:
            with closing(conn.cursor(pymysql.cursors.DictCursor)) as cur:
                for var_name, var_value in kwargs.items():
                    if (type(var_value) == dict) or (type(var_value) == list):
                        cur.execute(
                            "update %s set %s = %s where %s = %s" % (
                                table, var_name, conn.escape(json.dumps(var_value)), key, key_value))
                    else:
                        cur.execute(
                            "update %s set %s = %s where %s = %s" % (
                            table, var_name, conn.escape(var_value), key, key_value))
                conn.commit()