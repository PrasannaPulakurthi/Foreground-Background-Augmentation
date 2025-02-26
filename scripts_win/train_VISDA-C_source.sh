set CUBLAS_WORKSPACE_CONFIG=:4096:8
python main_win.py train_source=true learn=source seed=2022 port=10000 memo="source" project="VISDA-C" data.data_root="datasets" data.workers=8 data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" learn.epochs=10 model_src.arch="resnet101" optim.lr=2e-4
