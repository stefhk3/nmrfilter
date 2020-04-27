import os
import platform
from urllib.parse import quote_plus as urlquote

"""
Keep track of all database configuration information here. 
Access credentials should NOT EVER be checked into repo
instead source them from a shell script into environment vars
"""


AWS_RDS_USERNAME = os.environ.get('AWS_RDS_USERNAME', "")
AWS_RDS_PASSWORD = os.environ.get('AWS_RDS_PASSWORD', "")

AWS_RDS_DATABASE = "nmr"

# if platform.system() == "Darwin":
#     AWS_RDS_HOST = "localhost" # assume on mac we're port-forwarding 
# else:
AWS_RDS_HOST = "nmr-db-2.cau3ldtjxchh.us-west-2.rds.amazonaws.com"



AWS_RDS_DB_STR = 'postgresql://%s:%s@%s:5432/%s' % (urlquote(AWS_RDS_USERNAME), 
                                                    urlquote(AWS_RDS_PASSWORD), 
                                                    AWS_RDS_HOST, 
                                                    AWS_RDS_DATABASE)

