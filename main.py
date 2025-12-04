from utils.utils import get_articles_to_parse
from datetime import datetime
import time
from llm_tbl_generate import llm_table_generation
import yaml
with open('config.yaml', 'r') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
ts = datetime.now()
# get_articles_to_parse(conf, ts)
idx = 0
while True:
    try:
        llm_table_generation(conf, ts)
    except Exception as e:
        print(f'Error occurred: {e}, restarting...')
    time.sleep(30)
    idx += 1
    print(f'Iteration {idx} completed, restarting...')