"""Cross-platform, all training completes before any test evaluation."""
import argparse,subprocess,sys

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config/default.yaml'); p.add_argument('--seeds',type=int,nargs='+',default=[42]); p.add_argument('--prepare',action='store_true'); a=p.parse_args()
    def run(*args): subprocess.run([sys.executable,*args],check=True)
    if a.prepare: run('-m','src.data.preprocess')
    for seed in a.seeds:
        for model in ('rnn','gru','lstm','transformer'): run('-m','src.train','--model',model,'--config',a.config,'--seed',str(seed))
    run('-m','src.evaluate','--config',a.config,'--model','all','--seeds',*[str(s) for s in a.seeds])
    run('-m','src.figures','--config',a.config)
    run('-m','src.report','--config',a.config)
if __name__=='__main__': main()
