"""Evaluate a frozen training OCSVM on both 2018 benign and malware APKs."""
from modeling import build_features,evaluate_detector


if __name__=='__main__':
    from run_pipeline import main
    main(default_only='evaluate')
