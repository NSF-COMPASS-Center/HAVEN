#!/usr/bin/env python
#import numpy as np
#import random
#from Bio import Seq, SeqIO
from collections import Counter
from mutation import *
from keras.models import Sequential
from keras.models import Model
from tensorflow.keras import layers
from typing import List
import tensorflow as tf
import types
from AA import AAs

class Hep():
    def __init__(self, model_name:str, checkpoint:str=None, datasets:List[str]=None, transferCheckpoint:str=None,
            namespace:str="hep", seed:int=1, dim:int=512, batch_size:int=500, 
            n_epochs:int=11, train:bool=False, test:bool=False, embed:bool=False, 
            semantics:bool=False, combfit:bool=False, reinfection:bool=False, train_split:bool=False, visulise:bool=False):
        self.args = types.SimpleNamespace()
        self.args.model_name = model_name
        self.args.seed = seed
        self.args.namespace = namespace
        self.args.dim = dim
        self.args.batch_size = batch_size
        self.args.checkpoint = checkpoint
        self.args.n_epochs = n_epochs
        self.args.train = train
        self.args.train_split=train_split
        self.args.test = test
        self.args.embed = embed
        self.args.semantics = semantics
        self.args.combfit = combfit
        self.args.reinfection = reinfection
        self.args.datasets = datasets
        self.args.transferCheckpoint = transferCheckpoint
        self.args.visulise=visulise

        # Set seeds
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        tf.random.set_seed(self.args.seed)

        print(self.args)
 

    def parse_viprbrc(self, entry):
        fields = entry.split('|')
        if fields[7] == 'NA':
            date = None
        else:
            date = fields[7].split('/')[0]
            date = dparse(date.replace('_', '-'))

        country = fields[9]
        from locations import country2continent
        if country in country2continent:
            continent = country2continent[country]
        else:
            country = 'NA'
            continent = 'NA'

        from mammals import species2group

        # might error out
        if fields[8] in species2group:
            group = species2group[fields[8]]
        else:
            group = "NA"

        meta = {
            'strain': fields[5],
            'host': fields[8],
            'group': group,
            'country': country,
            'continent': continent,
            'dataset': 'viprbrc',
        'protein': fields[1],
        }
        return meta

    def parse_nih(self, entry):
        fields = entry.split('|')

        country = fields[3]
        from locations import country2continent
        if country in country2continent:
            continent = country2continent[country]
        else:
            country = 'NA'
            continent = 'NA'

        from mammals import species2group
        if fields[2] in species2group:
            group = species2group[fields[2]]
        else:
            group = "NA"
        if fields[2] == "":
            host = "unknown"
        else:
            host = fields[2]

        meta = {
            'strain': 'SARS-CoV-2',
            'host': host,
            'group': group,
            'country': country,
            'continent': continent,
            'dataset': 'nih',
        'protein': fields[1],
        }
        return meta

    def parse_gisaid(self, entry):
        fields = entry.split('|')

        type_id = fields[1].split('/')[1]

        if type_id in { 'bat', 'canine', 'cat', 'env', 'mink',
                        'pangolin', 'tiger' }:
            host = type_id
            country = 'NA'
            continent = 'NA'
        else:
            host = 'human'
            from locations import country2continent
            if type_id in country2continent:
                country = type_id
                continent = country2continent[country]
            else:
                country = 'NA'
                continent = 'NA'

        from mammals import species2group

        meta = {
            'strain': fields[1],
            'host': host,
            'group': species2group[host].lower(),
            'country': country,
            'continent': continent,
            'dataset': 'gisaid',
        }
        return meta


    def parse_manual(self,entry):
        fields = [x.strip() for x in entry.split('|')]

        country = 'NA'
        continent = 'NA'


        meta = {
            'protein' : fields[1],
            'host': fields[2],
            'group': fields[3],
            'country': country,
            'continent': continent,
            'dataset': 'viprbrc-genbank',
        }
        return meta



    # Seqs key:value | string sequence : [metadata1, metadata2, ...]
    def process(self, fnames):
        seqs = {}
        for fname in fnames:
            for record in SeqIO.parse(fname, 'fasta'):
                #if len(record.seq) < 1000:
                #    continue
                if str(record.seq).count('X') > 0:
                    continue
                if record.seq not in seqs:
                    seqs[record.seq] = []
                if "manual" in fname.lower():
                    meta = self.parse_manual(record.description)
                    if meta['host'] == "unknown":
                        continue
                elif "viprbrc" in fname.lower():
                    meta = self.parse_viprbrc(record.description)
                elif "ncbi" in fname.lower():
                    meta = self.parse_nih(record.description)
                meta['accession'] = record.description
                seqs[record.seq].append(meta)

        with open('data/hep/hep_all.fa', 'w') as of:
            for seq in seqs:
                metas = seqs[seq]
                for meta in metas:
                    of.write('>{}\n'.format(meta['accession']))
                    of.write('{}\n'.format(str(seq)))

        print(f"Dataset size: {len(seqs)}")
        return seqs

    def split_seqs(self, seqs, split_method='random'):
        train_seqs, test_seqs = {}, {}

        tprint('Splitting seqs...')
        for idx, seq in enumerate(seqs):
            if idx % 10 < 2:
                test_seqs[seq] = seqs[seq]
            else:
                train_seqs[seq] = seqs[seq]
        tprint('{} train seqs, {} test seqs.'
               .format(len(train_seqs), len(test_seqs)))

        return train_seqs, test_seqs

    def setup(self, args):
        fnames = self.args.datasets
        seqs = self.process(fnames)

        seq_len = max([ len(seq) for seq in seqs ]) + 2 # For padding
        vocab_size = len(AAs) + 2 # For padding characters

        model = get_model(self.args, seq_len, vocab_size,
                          inference_batch_size=self.args.batch_size)
        return model, seqs, seq_len, vocab_size

    def interpret_clusters(self, adata):
        clusters = sorted(set(adata.obs['louvain']))
        for cluster in clusters:
            tprint('Cluster {}'.format(cluster))
            adata_cluster = adata[adata.obs['louvain'] == cluster]
            for var in [ 'host', 'country', 'strain' ]:
                tprint('\t{}:'.format(var))
                counts = Counter(adata_cluster.obs[var])
                for val, count in counts.most_common():
                    tprint('\t\t{}: {}'.format(val, count))
            tprint('')

    def plot_umap(self, adata, categories, namespace='hep'):
        for category in categories:
            sc.pl.umap(adata, color=category,
                       save='_{}_{}.png'.format(namespace, category))

    def analyze_embedding(self, model, seqs, vocabulary):
        seqs = embed_seqs(self.args, model, seqs, vocabulary, use_cache=True)

        X, obs = [], {}
        obs['n_seq'] = []
        obs['seq'] = []
        for seq in seqs:
            meta = seqs[seq][0]
            X.append(meta['embedding'].mean(0))
            for key in meta:
                if key == 'embedding':
                    continue
                if key not in obs:
                    obs[key] = []
                obs[key].append(Counter([
                    meta[key] for meta in seqs[seq]
                ]).most_common(1)[0][0])
            obs['n_seq'].append(len(seqs[seq]))
            obs['seq'].append(str(seq))
        X = np.array(X)

        adata = AnnData(X)
        for key in obs:
            adata.obs[key] = obs[key]

        sc.pp.neighbors(adata, n_neighbors=20, use_rep='X')
        sc.tl.louvain(adata, resolution=1.)
        sc.tl.umap(adata, min_dist=1.)

        sc.set_figure_params(dpi_save=500)
        plot_umap(adata, [ 'host', 'group', 'continent', 'louvain' ])

        interpret_clusters(adata)

        adata_cov2 = adata[(adata.obs['louvain'] == '0') |
                           (adata.obs['louvain'] == '2')]
        plot_umap(adata_cov2, [ 'host', 'group', 'country' ],
                  namespace='cov7')

    def start(self):
        vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) }

        hosts = {"Human", "Non-Human"}
        hostVocab = {host : idx for idx, host in enumerate(sorted(hosts), start=1)}
        model, seqs, seq_len, vocab_size = self.setup(self.args)

        if self.args.visulise:
            print(f"visulise_dataset: {self.args.visulise}")
            # Extract freq map of species wide distribution
            # Basically a hashmap obj
            speciesCounter = Counter()

            # Count the hosts
            for sv in seqs.values():
                speciesCounter[sv[0]['host']]+=1

            speciesORF1Counter = Counter()
            for sv in seqs.values():
                if "orf1" in sv[0]["protein"].lower():
                    speciesORF1Counter[sv[0]['host']]+=1

            speciesORF2Counter = Counter()
            dataset = None
            for sv in seqs.values():
                if "orf2" in sv[0]["protein"].lower():
                    speciesORF2Counter[sv[0]['host']]+=1
                dataset = sv[0]["dataset"]

            # Vis with numpy
            import matplotlib.pyplot as plt
            def plotCounter(d, p_name):
                print(f"{p_name}: {d}")
                with plt.style.context("seaborn"):
                    #fig = plt.figure(1, [5, 4])
                    #plt.rcParams.update({'font.size': 22})
                    tmp = sorted(zip(d.keys(), d.values()), key=lambda x: -x[1])  
                    keys = [x[0] for x in tmp]
                    values = [y[1] for y in tmp]
                    plt.bar(keys, values)
                    plt.xticks(
                        rotation=45,
                        horizontalalignment='right',
                        #fontweight='heavy',
                        #fontsize='small'
                    )
                plt.xlabel('Host species')
                plt.ylabel('Sequence count')
                plt.title(f'Host species vs frequency count: ({dataset}, Orthohepevirus A, {p_name})')

                # Print image
                name = f'distribution-{p_name}.png'
                plt.savefig(name, bbox_inches='tight', dpi=300)
                plt.clf()

            # With all 3 frequency maps
            plotCounter(speciesCounter, 'overall')
            plotCounter(speciesORF1Counter, 'orf1')
            plotCounter(speciesORF2Counter, 'orf2')

        # If bilstm model has a checkpoint, load it in
        if self.args.checkpoint:
            model.model_.load_weights(self.args.checkpoint)
            tprint('Model summary:')
            tprint(model.model_.summary())
     
        # If bilstm model should train
        if self.args.train:
            batch_train(self.args, model, seqs, vocabulary, batch_size=self.args.batch_size)

        # If bilstm should train with splits or test
        if self.args.train_split or self.args.test:
            train_test(self.args, model, seqs, vocabulary, split_seqs)

        if self.args.embed:
            if self.args.checkpoint is None and not self.args.train:
                raise ValueError('Model must be trained or loaded '
                                 'from checkpoint.')
            no_embed = { 'hmm' }
            if self.args.model_name in no_embed:
                raise ValueError('Embeddings not available for models: {}'
                                 .format(', '.join(no_embed)))
            analyze_embedding(self.args, model, seqs, vocabulary)

        if self.args.semantics:
            if self.args.checkpoint is None and not self.args.train:
                raise ValueError('Model must be trained or loaded '
                                 'from checkpoint.')

            from escape import load_baum2020, load_greaney2020
            tprint('Baum et al. 2020...')
            seq_to_mutate, seqs_escape = load_baum2020()
            analyze_semantics(self.args, model, vocabulary,
                              seq_to_mutate, seqs_escape, comb_batch=10000,
                              prob_cutoff=0, beta=1., plot_acquisition=True,)
            tprint('Greaney et al. 2020...')
            seq_to_mutate, seqs_escape = load_greaney2020()
            analyze_semantics(self.args, model, vocabulary,
                              seq_to_mutate, seqs_escape, comb_batch=10000,
                              min_pos=318, max_pos=540, # Restrict to RBD.
                              prob_cutoff=0, beta=1., plot_acquisition=True,
                              plot_namespace='cov2rbd')

        # where to get data for this...
        if self.args.combfit:
            from combinatorial_fitness import load_starr2020
            tprint('Starr et al. 2020...')
            wt_seqs, seqs_fitness = load_starr2020()
            strains = sorted(wt_seqs.keys())
            for strain in strains:
                analyze_comb_fitness(self.args, model, vocabulary,
                                     strain, wt_seqs[strain], seqs_fitness,
                                     comb_batch=10000, prob_cutoff=0., beta=1.)

        if self.args.reinfection:
            from reinfection import load_to2020, load_ratg13, load_sarscov1
            from plot_reinfection import plot_reinfection

            tprint('To et al. 2020...')
            wt_seq, mutants = load_to2020()
            analyze_reinfection(self.args, model, seqs, vocabulary, wt_seq, mutants,
                                namespace='to2020')
            plot_reinfection(namespace='to2020')
            null_combinatorial_fitness(self.args, model, seqs, vocabulary,
                                       wt_seq, mutants, n_permutations=100000000,
                                       namespace='to2020')

            tprint('Positive controls...')
            wt_seq, mutants = load_ratg13()
            analyze_reinfection(self.args, model, seqs, vocabulary, wt_seq, mutants,
                                namespace='ratg13')
            plot_reinfection(namespace='ratg13')
            wt_seq, mutants = load_sarscov1()
            analyze_reinfection(self.args, model, seqs, vocabulary, wt_seq, mutants,
                                namespace='sarscov1')
            plot_reinfection(namespace='sarscov1')
