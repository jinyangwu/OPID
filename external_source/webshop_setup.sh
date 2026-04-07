conda create -n skillrl-webshop python==3.10 -y
conda activate skillrl-webshop

cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all

cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2

pip install -r requirements.txt;
pip install sentence-transformers faiss-cpu

pip install httpx[socks]