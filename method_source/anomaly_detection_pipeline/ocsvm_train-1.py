"""Train the checkpointed OCSVM using 2014-2017 benign APKs only."""
from modeling import build_features,train_detector


if __name__=='__main__':
    from run_pipeline import main
    main(default_only='train')
