set CUBLAS_WORKSPACE_CONFIG=:4096:8

python main_win.py train_source=true learn=source seed=2022 port=10001 memo="source" project="PACS" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[photo,art_painting,cartoon,sketch]" model_src.arch="resnet18" optim.lr=2e-4 data.batch_size=16

python main_win.py train_source=true learn=source seed=2022 port=10002 memo="source" project="PACS" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[photo,art_painting,cartoon,sketch]" model_src.arch="resnet18" optim.lr=2e-4 data.batch_size=16 learn.epochs=20

python main_win.py train_source=true learn=source seed=2022 port=10003 memo="source" project="PACS" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[cartoon]" data.target_domains="[photo,art_painting,cartoon,sketch]" model_src.arch="resnet18" optim.lr=2e-4 data.batch_size=16 learn.epochs=20

python main_win.py train_source=true learn=source seed=2022 port=10004 memo="source" project="PACS" data.data_root="datasets" data.workers=8 data.dataset="PACS" data.source_domains="[sketch]" data.target_domains="[photo,art_painting,cartoon,sketch]" model_src.arch="resnet18" optim.lr=2e-4 data.batch_size=16
