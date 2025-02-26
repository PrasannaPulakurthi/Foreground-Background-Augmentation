set CUBLAS_WORKSPACE_CONFIG=:4096:8

# P to A
python main_win.py seed=2022 port=10001 memo="target" project="PACS" learn.epochs=50 learn.patch_height=28 learn.num_neighbors=3 data.aug_type="mask" data.data_root="datasets" data.workers=4 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[art_painting]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4

# P to C
python main_win.py seed=2022 port=10002 memo="target" project="PACS" learn.epochs=50 learn.patch_height=28 learn.num_neighbors=3 data.aug_type="mask" data.data_root="datasets" data.workers=4 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4

# P to S
python main_win.py seed=2022 port=10003 memo="target" project="PACS" learn.epochs=50 learn.patch_height=28 learn.num_neighbors=3 data.aug_type="mask" data.data_root="datasets" data.workers=1 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[sketch]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4

# A to P
python main_win.py seed=2022 port=10004 memo="target" project="PACS" learn.epochs=25 learn.patch_height=28 learn.num_neighbors=3 data.aug_type="mask" data.data_root="datasets" data.workers=4 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[photo]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4

# A to C
python main_win.py seed=2022 port=10005 memo="target" project="PACS" learn.epochs=50 learn.patch_height=28 learn.num_neighbors=3 data.aug_type="mask" data.data_root="datasets" data.workers=4 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[cartoon]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 

# A to S
python main_win.py seed=2022 port=10006 memo="target" project="PACS" learn.epochs=50 learn.patch_height=28 learn.num_neighbors=3 data.aug_type="mask" data.data_root="datasets" data.workers=4 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[sketch]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4

# P to ACS
python main_win.py seed=2022 port=10007 memo="target" project="PACS" learn.epochs=50 learn.patch_height=28 learn.num_neighbors=3 data.aug_type="mask" data.data_root="datasets" data.workers=4 data.dataset="PACS" data.source_domains="[photo]" data.target_domains="[acs]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 data.ttd=true data.test_target_domain="[art_painting,cartoon,sketch]"

# A to PCS
python main_win.py seed=2022 port=10008 memo="target" project="PACS" learn.epochs=50 learn.patch_height=28 learn.num_neighbors=3 data.aug_type="mask" data.data_root="datasets" data.workers=4 data.dataset="PACS" data.source_domains="[art_painting]" data.target_domains="[pcs]" model_src.arch="resnet18" model_tta.src_log_dir="output/PACS/source" optim.lr=2e-4 data.ttd=true data.test_target_domain="[photo,cartoon,sketch]"


