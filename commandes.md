ssh -p 22222 -i C:\Users\aissi\.ssh\id_ed25519 student7@ictlab.usth.edu.vn

ssh ict14

screen -ls

screen -r ...

cd /storage/student7/teyssir/code

source .venv/bin/activate

python scripts/collect_cnn_results.py --output_dir /tmp/student7/output

cd /storage and mkdir

git merge origin/feature/cnn-ablation

# ============================================================
# RESULTATS CLES
# ============================================================
# Meilleur checkpoint : Step 9
# /storage/student7/teyssir/code/output/cnn_step9_geo_features/last.ckpt
#
# Threshold sweep Step 9 (point-level F1) :
#   0.50 → F1=89.27%  Prec=81.67%  Rec=98.44%   (défaut)
#   0.65 → F1=90.11%  Prec=83.99%  Rec=97.19%   (recommandé)
#   0.70 → F1=90.91%  Prec=90.71%  Rec=91.11%   (max F1, PTv3=90.94%)
#
# Low fracture ratio (<14%) : threshold optimal = 0.40
#
# Tableau ablation (fragment-level val F1) :
#   Step 7  U-Net            77.08%
#   Step 8b FeatFuse(max)    81.91%
#   Step 9  GeoFeatures      85.50%
#   PTv3 GARF                90.94%
#
# Params : Step 9 = 523K  vs  PTv3 = 12 727K  (24x moins)
# ============================================================
# Analyse erreurs Step 9
CUDA_VISIBLE_DEVICES=1 python scripts/analyze_errors.py \
    --ckpt /storage/student7/teyssir/code/output/cnn_step13_point_head/last-v3.ckpt \
    --data_root /storage/student7/teyssir/data/breaking_bad_vol.hdf5 \
    --experiment cnn_step13_point_head  \
    --out_dir /tmp/student7/analysis/step13_intermediate \
    --n_vis 6 --max_batches 300 --sweep_threshold
# Recuperer les figures d'analyse en local
scp -P 22222 -i C:\Users\aissi\.ssh\id_ed25519 \
    "student7@ictlab.usth.edu.vn:/tmp/student7/analysis/step9/*.png" \
    "C:\Users\aissi\Downloads\step9_analysis\"
    