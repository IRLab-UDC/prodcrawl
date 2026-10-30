# ProdCrawl

This repository contains the code to reproduce the experiments of ProdCrawl, a replayable test collection for focused crawling. 
It holds static mirrors of 621 company and organization websites and a relevance judgment for every page, for the predicate *"is this a product page?"*.

The repository includes:

- the focused crawler (Scrapy), with the baselines (BFS, DFS, SVM and Naive Bayes
  anchor-text classifiers) and the retrieval-based anchor scoring strategies (BM25,
  exact kNN, approximate kNN, and hybrid ranking);
- the embedding service
- the scripts to unpack and serve the mirrors.

The data are released separately:

| Resource | Location |
|---|---|
| Judgments and site mirrors (one WARC file per site) | [irlab-udc/prodcrawl](https://huggingface.co/datasets/irlab-udc/prodcrawl) |
| Fine-tuned anchor-text encoder | [irlab-udc/anchor-product-page-mpnet-base](https://huggingface.co/irlab-udc/anchor-product-page-mpnet-base) |

## Repository layout

```
focused_crawler/            Scrapy project: spider, scoring strategies, frontier, ES access
  spiders/ProductsSpider.py   the focused crawler
  common/scoring_strategies.py  link prioritisation strategies (baselines and retrieval-based)
  common/queues.py            priority frontier (min-heap, FIFO among equal priorities)
  common/link_extractors.py   link extraction
  common/db_utils.py          SQLite access: judgments and embedding caches
  index/embeddings_indexer.py Elasticsearch queries (BM25, kNN, aNN, hybrid)
  nlp/                        page text extraction and the trained classifier baselines
experiments/
  run_spiders_runner.py       runs one configuration: one crawler process per site
  launch_crawling_experiment.sh
  example_config.yaml         template for the configuration of a run
datasets/                   seed sites of the training, validation and test collections
mirror_server/              Apache server that serves the unpacked mirrors
nlp_api/                    embedding service (all-mpnet-base-v2 and the fine-tuned encoder)
scripts/data/               unpack the WARCs; build the SQLite databases
```

## Requirements

- Linux with Docker, to run the mirror server and Elasticsearch. Without Docker or
  root access, see "Running on a cluster without root".
- Python 3.10, in two virtual environments: one for the crawler and one for the
  embedding service. A GPU speeds up the embedding service but is not required.
- Elasticsearch 8.12.1, with `indices.query.bool.max_clause_count` set to 5000.
- Three local ports: one for the mirror server (80, or any port with the proxy setup of
  step 4), 9200 for Elasticsearch and 8030 for the embedding service.
- Disk space for the mirrors of the collections you crawl:

  | Collection | WARC files | Unpacked mirrors |
  |---|---:|---:|
  | Training | 15.0 GB | 81 GB |
  | Validation | 7.2 GB | 43 GB |
  | Test | 7.0 GB | 44 GB |

  The runs crawl the validation or test mirrors. The training mirrors are needed only
  to build the offline evidence stores, which take about 20 GB each in Elasticsearch.

## How a crawl works

A run crawls every site of one collection. The runner starts one crawler process per
site, up to 50 at a time. Each crawler starts at the site's home page on the mirror
server, stays within the site, and keeps one request in flight with a delay of 1 s.

For each downloaded page, the crawler:

1. looks up the page's judgment in `labels.db` (pages without a judgment count as
   non-relevant);
2. stores the page in the run's Elasticsearch index, with its download order, depth,
   anchor text and, for the dense strategies, the anchor-text embedding;
3. extracts the page's links and scores the first 100 internal links, in document
   order, with the configured strategy.

A link's score s ∈ [0, 1] becomes the frontier priority `round(1000 (1 − s))`. The
frontier pops the lowest priority first, and links with equal priority leave in the
order they were found. BFS and DFS use a counter as priority instead.

The retrieval-based strategies score a link by retrieving the k anchor texts most
similar to its own from an evidence store, and returning the similarity-weighted share
of those whose target page is relevant. There are three evidence stores:

- the **online** store: the run's own index, which grows during the crawl. The sites
  of a run are crawled in parallel and share it;
- the **offline** store: the labelled anchor texts of the training sites, built once
  by a breadth-first crawl (step 10);
- the **combined** store: both.

BFS, DFS and the classifier baselines are deterministic. From the released WARCs and
judgments, this code reproduces our BFS crawl of a test site page for page. Runs that
use the online store depend on how the parallel site crawls interleave, so repeated
runs can differ slightly.

## Running the experiments

This guide goes from a fresh machine to a first run on ten validation sites (steps 1
to 8), and then to the runs of the paper (steps 9 and 10). Commands run from the
repository root. Most steps end with a command that checks their result.

The crawler needs three services, which keep running while it crawls:

| Service | What it does | Started in |
|---|---|---|
| Mirror server (Apache) | serves the unpacked mirrors | step 4 |
| Elasticsearch | stores each run and the offline evidence stores | step 5 |
| Embedding service | embeds anchor texts with the two encoders | step 6 |

### 1. Install the crawler and the embedding service

The crawler and the embedding service pin different package versions, so each has its
own environment:

```bash
git clone https://github.com/jmanuelrey/prodcrawl-experiments.git
cd prodcrawl-experiments

# Crawler
python3.10 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt warcio==1.7.4

# Embedding service. Without a GPU, install the CPU build of PyTorch first:
# it is a much smaller download.
python3.10 -m venv .venv-nlp
.venv-nlp/bin/pip install "torch==2.12.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
.venv-nlp/bin/pip install -r nlp_api/requirements.txt
```

Install the crawler from `requirements-lock.txt`, which pins every package to the
version used in our runs. `requirements.txt` lists only the direct dependencies; on its
own it installs a w3lib release that Scrapy 2.6.2 cannot import. The embedding
service's environment also provides the `hf` command, used in the next step.

### 2. Download the data

For the first run, you can choose to download the judgments and the WARC files of ten validation sites
(about 300 MB):

```bash
awk 'NR % 17 == 9' datasets/validation.txt > first_run_sites.txt
.venv-nlp/bin/hf download irlab-udc/prodcrawl --repo-type dataset --local-dir ../prodcrawl-data \
  --include "judgments/*" $(sed 's|.*|--include warcs/validation/&.warc.gz|' first_run_sites.txt)
```

or either go for the entire collection:

```bash
.venv-nlp/bin/hf download irlab-udc/prodcrawl --repo-type dataset --local-dir ../prodcrawl-data
```

`datasets/validation.txt` lists the validation sites from the largest to the
smallest, so every 17th site gives ten sites of all sizes. For the runs of the paper,
download a whole collection with `--include "warcs/validation/*"` or
`--include "warcs/test/*"`, or omit `--include` to download everything.

### 3. Unpack the mirrors and build the databases

```bash
export MIRRORS=$HOME/prodcrawl-mirrors      # any directory with enough space
.venv/bin/python scripts/data/extract_warcs.py --out-dir $MIRRORS ../prodcrawl-data/warcs/validation/*.warc.gz
.venv/bin/python scripts/data/setup_databases.py --judgments ../prodcrawl-data/judgments/*.tsv --out-dir data
```

Each WARC record becomes the file `$MIRRORS/<host>/<path>` for the page
`http://<host>/<path>`, which is the URL used in the judgments. The ten first-run sites
unpack to 15,329 files (1.6 GB).

`setup_databases.py` writes `data/labels.db` with the 1,102,934 judgments, and two
empty embedding caches, `data/embeddings.db` and `data/embeddings_finetuned.db`. It
does not overwrite an existing `labels.db`.

### 4. Start the mirror server

Build the image. It runs Apache with the configuration used in our experiments:

```bash
docker build -t prodcrawl-mirror mirror_server/
```

The crawler requests the page `http://<host>/<path>` as `<MIRROR_URL><host>/<path>`,
and recovers the page's URL by removing the server part. `MIRROR_URL` must therefore
name the server without a port: with a port, the crawled URLs do not match the
judgments. There are two ways to meet this requirement.

**Port 80 is free.** Serve the mirrors on it:

```bash
docker run -d --name prodcrawl-mirror -p 127.0.0.1:80:80 -v $MIRRORS:/var/www/html:ro prodcrawl-mirror
export MIRROR_URL=http://127.0.0.1/
curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' http://127.0.0.1/lavcup.com
# 301 http://127.0.0.1/lavcup.com/
```

**Port 80 is taken or not available.** Serve the mirrors on another port, here 8080,
and let the crawler use the server as an HTTP proxy. The crawler then requests
`http://mirror/<host>/<path>`. The name `mirror` is never looked up, because every
request goes to the proxy, which serves it from the mirrors:

```bash
docker run -d --name prodcrawl-mirror -p 127.0.0.1:8080:80 -v $MIRRORS:/var/www/html:ro prodcrawl-mirror
export MIRROR_URL=http://mirror/
curl -s -o /dev/null -x http://127.0.0.1:8080 -w '%{http_code} %{redirect_url}\n' http://mirror/lavcup.com
# 301 http://mirror/lavcup.com/
```

Set the proxy only for the crawl (step 7), not for the whole shell. Otherwise every
HTTP request of the shell goes to the mirror server, including those of `pip` and
`hf`.

The server sets Content-Type from the file extension (Debian's `/etc/mime.types`) and
sends none for unknown extensions, such as `.php`. The crawler then detects HTML from
the body. Folders without `index.html` return a directory listing, as they did in our
experiments. Both setups return the same responses.

### 5. Start Elasticsearch

```bash
docker run -d --name prodcrawl-es -p 127.0.0.1:9200:9200 \
  -v prodcrawl-es-data:/usr/share/elasticsearch/data \
  -e discovery.type=single-node -e xpack.security.enabled=false \
  -e indices.query.bool.max_clause_count=5000 \
  -e "ES_JAVA_OPTS=-Xms16g -Xmx16g" elasticsearch:8.12.1
export ES_URL=http://localhost:9200
curl -s $ES_URL | grep '"number"'
#     "number" : "8.12.1",
```

Elasticsearch takes up to a minute to answer. Our runs used a 16 GB heap; a 6 GB heap
was enough for the first run. Security is disabled, so publish the port on 127.0.0.1
only, as above. The volume `prodcrawl-es-data` keeps the indices if the container is
removed.

### 6. Start the embedding service

The service loads two encoders when it starts: all-mpnet-base-v2 and the fine-tuned
`anchor-product-page-mpnet-base`. Download both once, before the first start:

```bash
.venv-nlp/bin/python -c "from sentence_transformers import SentenceTransformer as S; \
S('all-mpnet-base-v2'); S('irlab-udc/anchor-product-page-mpnet-base')"
```

Then start the service in a terminal of its own:

```bash
cd nlp_api
../.venv-nlp/bin/gunicorn nlp:app --workers 4 --worker-class uvicorn.workers.UvicornWorker \
  --bind 127.0.0.1:8030 --timeout 600
```

Keep `--timeout 600`. Each worker loads both encoders when it starts, and gunicorn's
default timeout of 30 s can kill a worker that is still downloading or loading them.
Gunicorn then starts a new worker, which meets the same fate: the log repeats
`[CRITICAL] WORKER TIMEOUT` and the service never answers.

Each worker holds its own copy of both encoders and uses the GPU when there is one.
`FINE_TUNED_MODEL` selects another copy of the fine-tuned encoder (a Hugging Face id
or a local path), and `HF_HOME` sets where Hugging Face stores the downloads.

Check the service from another terminal:

```bash
curl -s localhost:8030/status
# {"status":"OK"}
curl -s -X POST localhost:8030/get-embedding -H 'Content-Type: application/json' \
  -d '{"text_list": ["add to cart"], "model": "anchor-product-page-mpnet-base"}' | head -c 80
```

The configuration template expects the service at `http://localhost:8030`
(`nlp_api_url`).

### 7. First run: ten validation sites

Copy the configuration template and point it to the ten sites:

```bash
cp experiments/example_config.yaml first_run.yaml
sed -i 's|^seeds_path: .*|seeds_path: "first_run_sites.txt"|' first_run.yaml
```

The template runs exact kNN (`EmbeddingCosSim`) with k = 10, over the online store, with
the generic encoder and a budget of 252 responses per site. In the terminal of the
crawl, set `MIRROR_URL` and `ES_URL` as in steps 4 and 5, then run:

```bash
source .venv/bin/activate
./experiments/launch_crawling_experiment.sh --config first_run.yaml > first_run.log 2>&1
```

With the proxy setup of step 4, set the proxy for this command only, and exclude the
embedding service from it:

```bash
http_proxy=http://127.0.0.1:8080 no_proxy=localhost,127.0.0.1 \
  ./experiments/launch_crawling_experiment.sh --config first_run.yaml > first_run.log 2>&1
```

The log is verbose. On a machine with 8 CPUs and a GPU, this run took 5 minutes. It
writes the index
`embeddingcossim-top10-re_ranking0-coldstart-ratio-<commit>-first_run_sites-all-mpnet-base-v2`.

### 8. Check the run

```bash
grep -o "'finish_reason': '[a-z_]*'" first_run.log | sort | uniq -c
grep -c Traceback first_run.log
curl -s "$ES_URL/_cat/indices?v&h=index,docs.count"
curl -s "$ES_URL/*-first_run_sites-*/_search?size=0" -H 'Content-Type: application/json' \
  -d '{"aggs": {"sites": {"terms": {"field": "domain", "size": 1000},
       "aggs": {"relevant": {"filter": {"term": {"label": true}}}}}}}'
```

Our first run ended with no tracebacks and about 1,200 pages in the index. Four sites
stopped at the budget (`closespider_pagecount`) and six ran out of links to follow
(`finished`). The last command gives the pages and relevant pages downloaded per site.

Some responses have status 404: a few links point to files that the mirrors do not
hold. A 404 takes no position in the download order. It does count towards
`max_pages`, which limits responses.

Each run index holds one document per downloaded page, with its URL, its label from
the judgments, its depth and its download order per site. The paper evaluates a run by
MAP@b and AUMC, the area under the MAP@b curve for b = 1..30, with R_d counted in the
judgments of the collection (see the dataset card).

To run a configuration again, delete its index first:

```bash
curl -X DELETE "$ES_URL/<index>"
```

Otherwise the new run writes into the old index. Pages already there are not stored
again, and the online store starts with the evidence of the old run.

### 9. Run the experiments of the paper

A run of the paper crawls all 167 validation sites or all 136 test sites. For each run:

1. download and unpack the collection's WARC files (steps 2 and 3);
2. copy `experiments/example_config.yaml` and set `seeds_path`
   (`datasets/validation.txt` or `datasets/testing.txt`), `max_pages` (252 on
   validation, 265 on test), and the strategy and its parameters (see "Configuration");
3. for the offline and combined stores, build the stores first (step 10);
4. run it as in step 7.

Each run writes the index
`<strategy>-top<k>-re_ranking<w>-<store>-ratio-<commit>-<seeds>[-<model>]`, where
`<store>` is `coldstart` (online), `training_data` (offline) or
`coldstart__and__training_data` (combined), and `<seeds>` is the name of the seeds file
without its extension. Run from a git checkout: the commit hash is part of the index
name.

Our runs used a 64-CPU node, with Elasticsearch and the embedding service on other
machines. A run over the validation sites took a median of about 20 minutes with BM25
or the BM25→kNN hybrid, 3.5 to 6.5 hours with approximate kNN, and 9 hours with exact
kNN, which compares each link with every document of its evidence store.

### 10. Build the offline evidence stores

The offline store holds the labelled anchor texts of the training sites. Build it once,
with an exhaustive breadth-first crawl of the training sites. Unpack the training
mirrors (steps 2 and 3, with `--include "warcs/train/*"`), copy
`experiments/example_config.yaml` to `training_store.yaml`, and set

```yaml
seeds_path: "datasets/training.txt"
strategy: "BreadthFirstSearchScore"
ranking: 10
use_embeddings: "yes"
embeddings_model: "all-mpnet-base-v2"
max_pages: 0
max_links_per_page: 0
```

```bash
./experiments/launch_crawling_experiment.sh --config training_store.yaml
```

The crawl writes the index
`breadthfirstsearchscore-top10-re_ranking0-coldstart-ratio-<commit>-training-all-mpnet-base-v2`,
where `<commit>` is the commit of your checkout. This index is the offline store for
the generic encoder. For the fine-tuned encoder, run the same crawl again with

```yaml
embeddings_model: "anchor-product-page-mpnet-base"
clean_anchors: "yes"
embeddings_cache: "data/embeddings_finetuned.db"
```

which writes the same pages, embedded with that encoder, to
`breadthfirstsearchscore-top10-re_ranking0-coldstart-ratio-<commit>-training-anchor-product-page-mpnet-base`.

A run uses an offline store through its `start_data` field: the store's index name, or
`coldstart+<name>` for the combined store. Approximate kNN searches 1,000 candidates
per shard; in the paper it searched copies of the stores split into 80 shards, which
Elasticsearch's split API creates:

```bash
curl -X PUT "$ES_URL/<store>/_settings" -H 'Content-Type: application/json' \
  -d '{"index.blocks.write": true}'
curl -X POST "$ES_URL/<store>/_split/<store>-80" -H 'Content-Type: application/json' \
  -d '{"settings": {"index.number_of_shards": 80}}'
```

Our offline stores were built with an earlier version of the crawler, so a rebuilt
store may differ slightly from ours.

## Running on a cluster without root

Cluster nodes usually have no Docker and do not let users open port 80. We ran the
first run of step 7 on a Slurm node with Singularity 4.1: the three services and the
crawler on one node, the services on 127.0.0.1, and the mirror server used as a proxy
(step 4). Apptainer takes the same commands, with `apptainer` in place of
`singularity`. On a shared node, choose free ports, and set `ES_URL`, the proxy port
and `nlp_api_url` to match.

We kept the unpacked mirrors, the Elasticsearch data and the SQLite databases on the
node's local disk. Keep the SQLite databases there in any case: the crawler processes
share them through SQLite's file locks, which are not reliable on network file
systems. Set `L` to a directory there, for example
`L=/tmp/prodcrawl-$USER`, and unpack the mirrors into it in step 3, with
`MIRRORS=$L/mirrors`.

**Images.** Pull Elasticsearch from Docker Hub. Build the mirror image with Docker on
any machine, then convert it:

```bash
singularity pull es.sif docker://elasticsearch:8.12.1

docker build -t prodcrawl-mirror mirror_server/      # on a machine with Docker
docker save prodcrawl-mirror -o mirror.tar
singularity build mirror.sif docker-archive://mirror.tar
```

**Elasticsearch.** The image is read-only, but Elasticsearch writes its GC log and
heap dumps under its installation directory. Bind writable directories there; without
them it stops with `Error opening log file 'logs/gc.log': Read-only file system`.

```bash
mkdir -p $L/es-data $L/es-logs $L/es-tmp
singularity exec --bind $L:$L es.sif cp -r /usr/share/elasticsearch/config $L/es-config
ES_PATH_CONF=$L/es-config ES_TMPDIR=$L/es-tmp ES_JAVA_OPTS="-Xms16g -Xmx16g" \
singularity exec --bind $L:$L --bind $L/es-logs:/usr/share/elasticsearch/logs \
  --bind $L/es-data:/usr/share/elasticsearch/data es.sif /usr/share/elasticsearch/bin/elasticsearch \
  -Ediscovery.type=single-node -Expack.security.enabled=false \
  -Eindices.query.bool.max_clause_count=5000 \
  -Enetwork.host=127.0.0.1 -Ehttp.port=9200 \
  -Epath.data=$L/es-data -Epath.logs=$L/es-logs > $L/es.log 2>&1 &
export ES_URL=http://127.0.0.1:9200
```

**Mirror server.** Apache runs as your user on port 8080, with a copy of the image's
configuration. `ServerName mirror:80` makes its redirects (from `/<site>` to
`/<site>/`) point to `http://mirror/<site>/`, without the port.

```bash
P=8080
mkdir -p $L/apache-run $L/apache-lock $L/apache-logs
singularity exec --bind $L:$L mirror.sif cp -r /etc/apache2 $L/apache2
sed -i "s/^Listen 80$/Listen 127.0.0.1:$P/" $L/apache2/ports.conf
sed -i -e "s/<VirtualHost \*:80>/<VirtualHost *:$P>/" \
  -e "s/#ServerName www.example.com/ServerName mirror:80/" $L/apache2/sites-available/000-default.conf
echo "ServerName mirror:80" >> $L/apache2/apache2.conf
APACHE_RUN_USER="#$(id -u)" APACHE_RUN_GROUP="#$(id -g)" APACHE_PID_FILE=$L/apache-run/apache2.pid \
APACHE_RUN_DIR=$L/apache-run APACHE_LOCK_DIR=$L/apache-lock APACHE_LOG_DIR=$L/apache-logs \
singularity exec --bind $L:$L --bind $L/mirrors:/var/www/html mirror.sif \
  /usr/sbin/apache2 -d $L/apache2 -f $L/apache2/apache2.conf -D FOREGROUND > $L/apache.log 2>&1 &
export MIRROR_URL=http://mirror/
curl -s -o /dev/null -x http://127.0.0.1:$P -w '%{http_code} %{redirect_url}\n' http://mirror/lavcup.com
# 301 http://mirror/lavcup.com/
```

**Embedding service.** Start it as in step 6, from a Python environment on the node.
To use the GPU from inside an image, run it with `singularity exec --nv`.

**Crawler.** Run it as in step 7, with the proxy variables. If the launch script
cannot read the commit hash (the checkout is not a git repository, or git reports
`detected dubious ownership`), pass it by hand with `--commit-hash <hex hash>`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ImportError: cannot import name '_safe_chars' from 'w3lib.util'` | The crawler was installed from `requirements.txt`. Install `requirements-lock.txt` (step 1). |
| The embedding service log repeats `[CRITICAL] WORKER TIMEOUT` and `/status` never answers | Gunicorn kills the workers while they load the encoders. Start it with `--timeout 600`, after downloading the encoders (step 6). |
| `KeyError: 'getpwuid(): uid not found'` when the embedding service starts in a container | The container runs under a user ID without a user name, for example with `docker run --user`. Set the variables `USER` and `LOGNAME`, for example `-e USER=crawler -e LOGNAME=crawler`. |
| Docker cannot start the mirror server: port 80 `is already allocated` or `address already in use` | Another program uses port 80. Use another port and the proxy setup (step 4). |
| The crawl logs `DNS lookup failed: no results for hostname lookup: mirror` | `MIRROR_URL` is `http://mirror/` but the crawl runs without the proxy. Set `http_proxy` for the crawl (step 7). |
| With the proxy, the requests to the embedding service fail | They went to the mirror server. Add the host of `nlp_api_url` to `no_proxy` (step 7). |
| `data/labels.db already exists; remove it first` | `setup_databases.py` does not overwrite the judgments database. Remove `data/labels.db` and run it again. |
| `ERROR: run from a git checkout or pass --commit-hash <hex hash>` | The launch script could not read the commit hash. Pass `--commit-hash`. |
| A repeated run adds few pages to its index | The index of the earlier run still exists. Delete it before the run (step 8). |
| Under Singularity, Elasticsearch stops with `Error opening log file 'logs/gc.log': Read-only file system` | Bind writable directories on its `logs` and `data` directories (see "Running on a cluster without root"). |

## Configuration

A run is described by one YAML file; `experiments/example_config.yaml` is a commented
template. Fields:

| Field | Meaning |
|---|---|
| `seeds_path` | sites to crawl: `datasets/training.txt`, `validation.txt` or `testing.txt` |
| `strategy` | scoring class in `focused_crawler/common/scoring_strategies.py` |
| `ranking` | k, the number of anchor texts retrieved |
| `re_ranking` | window of the BM25 first stage in the hybrid |
| `start_data` | evidence store: `coldstart` (online), the index name of an offline store, or both joined by `+` |
| `embeddings_model` | `all-mpnet-base-v2` or `anchor-product-page-mpnet-base` |
| `use_embeddings` | store anchor-text embeddings in the run index (needed by the dense strategies) |
| `clean_anchors` | lowercase and normalise whitespace before embedding |
| `score_option` | `ratio`: similarity-weighted share of relevant neighbours |
| `max_pages` | responses per site before the crawl stops (0: no limit) |
| `max_links_per_page` | links scored per downloaded page (default 100; 0: no limit) |
| `max_concurrent_spiders` | sites crawled in parallel |
| `test` | 1: crawl the mirrors at `MIRROR_URL`; 0: crawl the live web |
| `anchor_window` | 0: anchor text only |
| `db_dir`, `embeddings_cache`, `nlp_api_url` | judgments database, embedding cache, embedding service |

Settings of the experiments reported in the paper:

| Approach | `strategy` | `ranking` (k) | `re_ranking` (w) | Stores |
|---|---|---|---|---|
| BFS | `BreadthFirstSearchScore` | | | |
| DFS | `DepthFirstSearchScore` | | | |
| TF-IDF + SVM | `SVMClassifierScore` | | | |
| TF-IDF + NB | `NaiveBayesClassifierScore` | | | |
| BM25 | `BM25` | 5, 10, 15, 20 | | online, offline |
| exact kNN | `EmbeddingCosSim` | 5, 10, 15, 20, 50 | | online, offline, combined |
| approximate kNN | `EmbeddingCosSimAproximated` | 5, 10, 15, 20, 50 | | online, offline, combined (80-shard copies) |
| BM25→kNN hybrid | `BM25EmbeddingCosSim` | 5, 10, 15, 20, 50 | 50, 100, 150, 200, 500 | online, offline, combined |

- The dense strategies (kNN, approximate kNN, hybrid) ran with each encoder:
  `all-mpnet-base-v2` with `clean_anchors: "no"` and `data/embeddings.db`, and
  `anchor-product-page-mpnet-base` with `clean_anchors: "yes"` and
  `data/embeddings_finetuned.db`. They set `use_embeddings: "yes"`; the other
  approaches set `"no"`.
- The hybrid skipped k = w = 50.
- All runs used `score_option: "ratio"`, `anchor_window: 0`, `max_links_per_page: 100`
  and `max_concurrent_spiders: 50`.

## Trained models

The TF-IDF + SVM (C = 0.1, 1,000 iterations) and TF-IDF + Naive Bayes baselines use the
trained classifiers in `focused_crawler/nlp/models/`. The fine-tuned encoder,
`anchor-product-page-mpnet-base`, fine-tunes all-mpnet-base-v2 on the anchor texts of
the training sites, so that anchors of product pages lie closer to each other and
further from the anchors of other pages. The embedding service downloads it from
Hugging Face.

## License

This code is released under the
[Creative Commons Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/);
see [LICENSE](LICENSE). The ProdCrawl judgments and mirrors and the fine-tuned encoder
are released on Hugging Face under the same licence.
