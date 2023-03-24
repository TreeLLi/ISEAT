


from config.evaluation import *

from src.utils.helper import run

def evaluate(args):
    for log_id in args.log_ids:
        log = args.logger.fetch(log_id)
        

if __name__ == '__main__':
    config = EvaluationConfig()
    run(evaluate, cfg)
