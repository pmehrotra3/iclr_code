"""pages.py — the words of the preview pages. build.py puts each note beside the code it explains.

A page is a dict: slug (file name), short (nav label), title, kicker, lede, intro (HTML), files.
A file is {path, about, intro_title, intro, notes}; notes = [(anchor, title, HTML)] where a note
starts at the first line (after the previous note) whose text begins with `anchor`. title None
means "use the function / class name". Every line of every file is shown; pieces without a
note of their own belong to the note above them.
"""
from __future__ import annotations

import diagrams

# =============================================================================================
# start page
# =============================================================================================
INDEX_INTRO = """
<h2>The question</h2>
<p>A diffusion model makes a sample by starting from pure random noise and removing that noise
step by step. In this code the removal is <b>deterministic</b> (the DDIM sampler, or an ODE
solver for flow matching): the same starting noise always gives the same sample. So each starting
point, called a <b>seed</b>, has a <b>fate</b> fixed in advance. It either becomes one of the
kinds of data the model was trained on, or it lands somewhere no real data lives. That second case
is a <b>hallucination</b>.</p>
<p>The code asks: <i>can we tell a seed's fate just by looking at the seed, without running the
model?</i> A map of seed space that answers this is called the <b>atlas</b>.</p>

<h2>The toy world that makes it answerable</h2>
<p>With real images you never know the right answer, so the data here is a simple, fully known
mixture: <b>K round Gaussian blobs</b> (the <b>modes</b>) in <b>d</b> dimensions, their centres
spread over a sphere and kept far apart. Each blob has a <b>99% ball</b> of radius <b>R99</b> that
holds 99% of its points. That gives an exact rule for a fate
(<a href="core.html#label_fate"><code>core.label_fate</code></a>):</p>
<div class="formula">endpoint within R99 of its nearest mode centre k   →  fate = k   (0 … K−1)
otherwise                                         →  fate = −1  (hallucination)</div>
<p>Because the data is a Gaussian mixture, the <b>exact score</b> (the direction a perfect
denoiser would push each point) has a formula. So the code has <b>two samplers</b>:</p>
<ul>
<li><b>The learned sampler</b>: a small neural network trained on samples of the mixture, as in
real life (<a href="train.html">train.py</a>). This is the thing being studied.</li>
<li><b>The exact sampler</b>: the formula itself, no training, correct up to its step size
(<a href="core.html#true_score">core.py</a>). It can also run <i>backwards</i>, from data to
seed, and that is what makes the atlas possible.</li>
</ul>

<figure>{{FATE_MAP}}
<figcaption>{{FATE_CAPTION}}</figcaption></figure>

<h2>How the atlas is built and scored</h2>
<div class="flow">
<div class="stage"><div class="n">step 1</div><b>Plant labelled anchors</b><p>Points around every
mode in <i>data</i> space: inside the R99 ball (label k) and in a thin band just outside it
(label −1).</p></div>
<div class="arrow">→</div>
<div class="stage"><div class="n">step 2</div><b>Carry them back</b><p>Run the exact sampler backwards,
data → seed. Now we know which seeds lead to each label.</p></div>
<div class="arrow">→</div>
<div class="stage"><div class="n">step 3</div><b>Fit predictors</b><p>kNN, altered kNN, quadratic and
polar classifiers learn "seed → fate" from those pairs.</p></div>
<div class="arrow">→</div>
<div class="stage"><div class="n">steps 4–5</div><b>Score against the learned model</b><p>Fresh seeds go
through the <i>learned</i> sampler. How often did each predictor guess their fate right?</p></div>
</div>
<p>All of this is <a href="evaluate.html">evaluate.py</a>; the anchors are drawn by
<a href="core.html#ball_anchors">core.ball_anchors</a> and the predictors live in
<a href="fate.html">fate.py</a>.</p>

<h2>The pipeline: what runs when you type <code>python code/main.py</code></h2>
<div class="flow">
<div class="stage"><div class="n">stage 1 · train.py</div><b>Train learned samplers</b>
<p>One network per (d, K, repeat seed). Also runs it on the evaluation seeds once and caches their
fates (the ground truth).</p>
<div class="out">→ checkpoints/&lt;process&gt;/&lt;variant&gt;/</div></div>
<div class="arrow">→</div>
<div class="stage"><div class="n">stage 2 · evaluate.py</div><b>Build and score the atlas</b>
<p>Steps 1–5 above, for every anchor budget and predictor. One small JSON file per cell.</p>
<div class="out">→ output/&lt;run_id&gt;/…/T&lt;T_true&gt;/cells/</div></div>
<div class="arrow">→</div>
<div class="stage"><div class="n">merge · evaluate.py</div><b>Collect</b>
<p>All cell files → results.json and two CSV tables, mean ± std over the repeat seeds.</p>
<div class="out">→ results.json, results.csv</div></div>
<div class="arrow">→</div>
<div class="stage"><div class="n">stage 3 · visualize.py</div><b>Draw</b>
<p>Heatmaps over the (d, K) grid and a LaTeX / PNG table.</p>
<div class="out">→ table.tex, table.png, &lt;model&gt;.png</div></div>
</div>
<p><a href="main.html">main.py</a> reads the settings from <a href="config.html">conf/</a>, checks
them against the run's earlier settings (<a href="main.html#file-code-runstate-py">runstate.py</a>),
then runs the stages in order. For the full grid on every GPU there is
<a href="scripts.html">scripts/main.sh</a>.</p>

<h2>Map of the code</h2>
<div class="cards">{{CARDS}}</div>

<h2>A good reading order</h2>
<ol>
<li>This page, then <a href="core.html#label_fate"><code>label_fate</code></a> and
<a href="core.html#true_score"><code>true_score</code></a> in core.py: the two ideas everything rests on.</li>
<li><a href="evaluate.html#eval_one"><code>evaluate.eval_one</code></a>: the experiment itself, in one function.</li>
<li><a href="fate.html">fate.py</a>: the predictors and the scores.</li>
<li><a href="train.html#train_cell"><code>train.train_cell</code></a>: where the learned samplers come from.</li>
<li>Everything else is plumbing: settings, resuming, several machines, figures.</li>
</ol>

<h2>Glossary</h2>
<div class="tablewrap"><table>
<tr><th>word</th><th>meaning in this code</th></tr>
<tr><td><b>d</b></td><td>the dimension of the space (2 … 512).</td></tr>
<tr><td><b>K</b></td><td>the number of modes (blobs) in the mixture (2 … 32).</td></tr>
<tr><td><b>mode</b></td><td>one Gaussian blob N(μ<sub>k</sub>, σ²I). σ = 0.1; the centres μ<sub>k</sub> sit on a sphere of radius 2·√(d/2).</td></tr>
<tr><td><b>R99</b></td><td>radius of the ball holding 99% of one mode's mass. Grows like σ√d. The line between "a mode" and "a hallucination".</td></tr>
<tr><td><b>seed</b> (noise)</td><td>a starting point x ~ N(0, I) that a sampler turns into a sample.</td></tr>
<tr><td><b>fate</b></td><td>where a seed ends up: mode k (0 … K−1), or −1 = hallucination.</td></tr>
<tr><td><b>score / exact field</b></td><td>the direction that increases data probability fastest. Known exactly for a Gaussian mixture; for flow matching the equivalent is the exact <i>velocity</i>.</td></tr>
<tr><td><b>process</b></td><td><code>ddim</code> (diffusion, DDIM sampler) or <code>flow</code> (flow matching, optimal-transport path).</td></tr>
<tr><td><b>T_train</b></td><td>number of sampling steps of the learned sampler (500).</td></tr>
<tr><td><b>T_true</b></td><td>number of steps of the exact sampler used to build the atlas (100, 200 or 500). Results go to a folder <code>T&lt;T_true&gt;</code>.</td></tr>
<tr><td><b>anchor</b></td><td>a labelled point planted around a mode in data space, then carried back to seed space.</td></tr>
<tr><td><b>budget</b> (<code>n_per_mode</code>)</td><td>how many anchors per mode: 20 000, 50 000 or 100 000.</td></tr>
<tr><td><b>predictor</b></td><td>a model of "seed → fate": knn, altered_knn, quadratic, polar3. The config group is called <code>classifier</code>.</td></tr>
<tr><td><b>ground truth</b></td><td>the fates of fresh seeds under the <i>learned</i> sampler: 50 000 × K seeds per cell.</td></tr>
<tr><td><b>cell</b></td><td>one (d, K, repeat seed) combination; results are stored per cell.</td></tr>
<tr><td><b>run / run_id</b></td><td>a named experiment; everything it produces lives in <code>output/&lt;run_id&gt;/</code>.</td></tr>
<tr><td><b>variant</b></td><td><code>unweighted</code> (all modes equally likely) or <code>weighted</code> (random mixing weights).</td></tr>
</table></div>

<div class="callout warn">
<p><b>Three names that mean two things</b></p>
<ul>
<li><b>seed</b> is both a <i>noise vector</i> (a starting point x) and an <i>integer RNG seed</i>
for a repeat of the experiment (0, 100, 200). "n_seeds = 3" means three repeats, not three noise vectors.</li>
<li><b>classifier</b> in <code>conf/classifier/</code> means the <i>fate predictors</i>, not an image
classifier.</li>
<li><b>T</b> is a number of steps (T_train, T_true), and inside the samplers <code>i</code> or
<code>t</code> is the current step.</li>
</ul></div>
"""

CARDS = [
    ("main", "code/main.py, code/runstate.py", "Starts everything; checks a run's settings stay the same."),
    ("config", "conf/, README, requirements", "Every setting, and what changing it does."),
    ("core", "code/core.py", "The maths: the mixture, the network, training, the exact sampler, anchors."),
    ("processes", "code/processes/", "DDIM and flow matching behind one interface."),
    ("train", "code/train.py", "Stage 1: train the learned samplers, cache their fates."),
    ("evaluate", "code/evaluate.py", "Stage 2: build the atlas, score it, collect the results."),
    ("fate", "code/fate.py", "The four predictors and the accuracy / F1 scores."),
    ("visualize", "code/visualize.py", "Stage 3: heatmaps and tables from results.json."),
    ("combine", "code/combine.py", "Merge a run computed on another machine."),
    ("rank_probe", "code/rank_probe.py", "Stand-alone: do hallucinating seeds live in a low-dimensional set?"),
    ("scripts", "scripts/", "The full sweep on all GPUs, or on a SLURM cluster."),
    ("tests", "code/unit_tests/", "Proof that resuming and merging give identical results."),
]

# =============================================================================================
# main.py + runstate.py
# =============================================================================================
MAIN_FILES = [
    {"path": "code/main.py",
     "intro_title": "What this file is",
     "intro": """<p>The single entry point. You never import it; you run it:</p>
<div class="formula">python code/main.py                         # the sweep in conf/
python code/main.py sweep.d=[16,64]          # change any setting on the command line
python code/main.py stages=[visualize]       # only redraw the figures</div>
<p>Line 18 sets a PyTorch memory option <i>before</i> torch is imported (it has no effect
afterwards). The evaluation stage holds a few very large anchor tensors, and this option stops GPU
memory from fragmenting around them.</p>
<p>Line 25 lets the stage modules import each other by plain name (<code>import core</code>)
however the script is started.</p>""",
     "notes": [
         ("STAGES = {", "The list of stages",
          """<p>A stage is just a function that takes the whole configuration. <code>stages=[...]</code>
on the command line picks which ones run and in which order. <code>merge</code> is a separate stage
so that <a href="scripts.html">scripts/main.sh</a> can run many evaluation jobs in parallel and
then combine their files once.</p>"""),
         ("@hydra.main", "main(): load settings, check the run, run the stages",
          """<p><b>Hydra</b> builds the configuration: it reads <code>conf/config.yaml</code>, pulls in
one file from each group (process, data, train, …) and applies your command-line overrides. The
result is one nested object, <code>cfg</code>, passed to every stage.</p>
<ol>
<li>Print the full configuration (so every log file records what was run).</li>
<li>Reject unknown stage names <i>before</i> doing any work.</li>
<li><a href="#check_and_record"><code>runstate.check_and_record</code></a> makes sure this
invocation computes things the same way as earlier invocations of the same <code>run_id</code>,
and logs it.</li>
<li>Run each stage.</li>
</ol>"""),
     ]},
    {"path": "code/runstate.py",
     "intro_title": "Why runs have names",
     "intro": """<p>A big sweep takes days and is often done in pieces: d = 2…64 today, d = 128 tomorrow,
another machine for d = 512, a new predictor next week. All the pieces write into the same
folder, <code>output/&lt;run_id&gt;/</code>, and every stage skips work that is already there.</p>
<p>That is only safe if every piece computes a cell <i>the same way</i>. So the first invocation
writes its settings to <code>run.json</code>, and later ones are compared against it. Settings that
only choose <i>which</i> cells to compute (d, K, how many seeds, the device…) may differ; settings
that change a result (training steps, learning rate, a predictor's k…) may not.</p>""",
     "notes": [
         ("_SELECT = ", "Which settings may differ between pieces",
          """<ul>
<li><code>_SELECT</code>: top-level keys that only choose what to compute or where. Ignored
completely.</li>
<li><code>_DROP</code>: per-invocation switches such as <code>force_retrain</code>. Ignored wherever
they appear in the tree.</li>
</ul>
<p>Everything else is part of the run's identity.</p>"""),
         ("def run_dir", None, "<p>The run folder: <code>output/&lt;run_id&gt;</code>.</p>"),
         ("def _strip", None, "<p>Removes the <code>_DROP</code> keys from a nested dict, at any depth.</p>"),
         ("def identity", "identity(): the settings a run is made of",
          """<p>Turns the configuration into the record stored in <code>run.json</code>, in three parts:</p>
<ul>
<li><b>config</b>: data, training, evaluation, anchors… (minus <code>_SELECT</code>).
<code>data.weighted</code> is removed because the weighted and unweighted variants are allowed in
the same run (they go to different folders).</li>
<li><b>process</b>: the DDIM or flow settings, keyed by process name, with <code>T_true</code>
removed (different T_true values also go to different folders).</li>
<li><b>predictors</b>: each predictor's final settings (shared defaults + its own).</li>
</ul>
<p>Keying process and predictors by name is what lets you <i>add</i> flow or a new predictor to an
existing run later.</p>"""),
         ("def _diff", "_diff() and _merge(): compare and extend",
          """<p><code>_diff</code> walks two nested dicts and lists every value that both have but that
differs. A key present on only one side (a new predictor, a setting added to the code later) is
<b>not</b> a difference. <code>_merge</code> keeps the old record and adds whatever only the new one
has.</p>"""),
         ("def settings_diff", None, """<p>Applies <code>_diff</code> to the three parts and labels
each difference, for example <code>config.train.base_steps</code>.</p>"""),
         ("def check_and_record", "check_and_record(): the gatekeeper",
          """<ol>
<li>Open <code>run.json</code> with an exclusive <b>file lock</b>, because scripts/main.sh starts many
jobs at the same moment and they must not overwrite each other.</li>
<li>If the file already has settings and any differ, stop with a message that lists each one:
<div class="formula">run 'abc123' was made with different settings:
    config.train.base_steps: run has 30000, this invocation 151
  -&gt; use a new run_id, or strict_run=false to mix on purpose</div></li>
<li>Otherwise save the merged record (adding new processes or predictors) with updated time stamps.</li>
<li>Save this invocation's full configuration in <code>invocations/</code> and append one line
to <code>history.log</code>.</li>
</ol>"""),
     ]},
]

# =============================================================================================
# core.py
# =============================================================================================
CORE_FILE = {
    "path": "code/core.py",
    "intro_title": "What lives in core.py",
    "intro": """<p>All the numerical work that more than one stage needs, so that training and
evaluation use <i>exactly</i> the same maths. The file has six parts, shown in its docstring:
devices and repeat seeds; the Gaussian mixture; the network; the training recipe; the exact
sampler; the anchors.</p>
<p><code>scipy.stats.chi2</code> is the only non-PyTorch maths import; it gives R99.</p>""",
    "notes": [
        ("SEED_STRIDE", "Devices and repeat seeds",
         """<p><code>get_device</code> picks the GPU if one is visible, unless you say
<code>device=cpu</code>.</p>
<p><code>seed_list</code> gives the repeat seeds: with <code>seed=0, n_seeds=3</code> that is
<b>0, 100, 200</b>. Each repeat puts the modes in new places, initialises the network differently
and uses different evaluation seeds, so the spread over repeats is honest.</p>
<p>Why steps of 100? Inside one repeat, each <i>purpose</i> gets its own small offset, and a stride
of 100 guarantees two repeats never share a random stream:</p>
<div class="tablewrap"><table>
<tr><th>offset</th><th>used for</th><th>where</th></tr>
<tr><td class="num">+0</td><td>mode placement, network initialisation</td><td>sample_modes, learned_closure</td></tr>
<tr><td class="num">+1</td><td>the evaluation seeds (ground truth)</td><td>train._save_cell, evaluate</td></tr>
<tr><td class="num">+2</td><td>calibration seeds for altered_knn</td><td>evaluate.eval_one</td></tr>
<tr><td class="num">+7</td><td>probe seeds (is training good enough?)</td><td>train.train_cell</td></tr>
<tr><td class="num">+11</td><td>the fixed training set</td><td>gmm_train_set</td></tr>
<tr><td class="num">+13, +17</td><td>mixing-weight profile, and its shuffle</td><td>mode_weights</td></tr>
<tr><td class="num">+29</td><td>every random draw during training</td><td>step_generator</td></tr>
</table></div>"""),
        ("def make_schedule", "make_schedule(): how much noise at each step",
         """<p>Diffusion adds noise gradually. At step i a clean point x₀ has become</p>
<div class="formula">x_i = √ᾱ_i · x₀ + √(1 − ᾱ_i) · noise</div>
<p>where ᾱ ("alpha-bar") falls from almost 1 (clean) to near 0 (all noise). The betas
β<sub>i</sub> are small per-step noise amounts, spaced evenly from <code>beta_min</code> = 10⁻⁴ to
<code>beta_max</code> = 0.02, and ᾱ<sub>i</sub> = (1−β₀)(1−β₁)…(1−β<sub>i</sub>).</p>
<div class="callout warn"><p><b>Worth knowing: the total amount of noise depends on T.</b>
The betas are spread between the same two end values whatever T is, so fewer steps means less noise
in total. At the seed end:</p>
<div class="tablewrap"><table>
<tr><th class="num">T</th><th class="num">ᾱ at the seed end</th><th class="num">data signal left in a "seed"</th></tr>
<tr><td class="num">100</td><td class="num">0.36</td><td class="num">60%</td></tr>
<tr><td class="num">200</td><td class="num">0.13</td><td class="num">36%</td></tr>
<tr><td class="num">500</td><td class="num">0.006</td><td class="num">8%</td></tr>
<tr><td class="num">1000</td><td class="num">0.00004</td><td class="num">0.6%</td></tr>
</table></div>
<p>The learned sampler uses T_train = 500; the exact sampler that builds the atlas uses T_true
(200 by default). For <b>DDIM</b>, when the two differ, the exact backtrack and the learned sampler
take the seed to be noise at <i>different</i> noise levels. The atlas then describes a slightly
different seed space from the one it is scored on. Flow matching is not affected: its time grid
always runs from 0 to 1. (This is an observation from reading the code, not something the code
states.)</p></div>"""),
        ("def r99", "r99(): the radius that holds 99% of a mode",
         """<p>For a d-dimensional Gaussian with spread σ, the squared distance from the centre divided
by σ² follows a <b>chi-square</b> distribution with d degrees of freedom. Its 99th percentile gives
the radius:</p>
<div class="formula">R99 = σ · √(chi2.ppf(0.99, d))</div>
<p>With σ = 0.1: R99 = 0.30 at d = 2, 0.97 at d = 64, 2.43 at d = 512. In high d almost all the
mass sits in a thin shell just inside R99, which is why small errors in high d push samples out of
the ball.</p>"""),
        ("def sample_modes", "sample_modes(): where the K blobs go",
         """<p>Draws K random directions, scales each to length R (the sphere radius) and accepts a
new centre only if it is at least <code>m_mult × 2 × R99</code> away from every centre already
placed. With m_mult = 1.5 the 99% balls never touch and have a gap between them. If it cannot fit K
centres after 10 000·K tries it gives up with a clear error; <a href="train.html#ModePlacementError">train.py</a>
then skips that cell.</p>
<p>Uses NumPy's old <code>RandomState</code> so that the placement is identical on every machine and
device.</p>"""),
        ("WEIGHT_LO", "mode_weights(): equal or unequal blobs",
         """<p>Unweighted (the default): every mode has probability 1/K.</p>
<p>Weighted (<code>data.weighted=true</code>): draw K numbers between 0.3 and 0.8 and normalise them,
<b>once</b> per K. Each repeat seed then hands these same weights to the modes in its own shuffled
order. So the spread across repeats comes from which mode gets which weight, not from new weights.
Always drawn on the CPU so that GPU and CPU runs agree.</p>"""),
        ("def _draw_modes", "Training data: _draw_modes, gmm_train_set, _minibatch",
         """<ul>
<li><code>_draw_modes</code>: pick a mode for each sample (uniformly, or by the weights).</li>
<li><code>gmm_train_set</code>: a fixed training set of <code>n_train</code> = 100 000 points,
μ<sub>k</sub> + σ·noise, reproducible from the seed.</li>
<li><code>_minibatch</code>: returns a function <code>draw()</code> that gives one batch per call,
either resampled from that fixed set or, if <code>n_train</code> is null, fresh from the mixture
every time. All randomness comes from the generator <code>g</code> passed in (see
<a href="#step_generator">step_generator</a>).</li>
</ul>"""),
        ("def label_fate", "label_fate(): the rule that defines a hallucination",
         """<p>The most important function in the repository. For each endpoint:</p>
<ol><li>measure the distance to every mode centre (<code>torch.cdist</code>);</li>
<li>take the nearest;</li>
<li>if it is within R99 the fate is that mode's index, otherwise −1.</li></ol>
<p>It works in chunks of 50 000 points so that the distance matrix never gets too big for memory.
Everything in the project (the ground truth, the anchor labels, the probe during training) is
labelled by this one function.</p>"""),
        ("class SinusoidalPosEmb", "SinusoidalPosEmb: telling the network the time",
         """<p>The network needs to know how noisy its input is. The step number t is turned into a
vector of sines and cosines at many frequencies (the same trick transformers use for word
positions), so that nearby steps get similar vectors and the network can tell steps apart
easily.</p>"""),
        ("class MLPBlock", "MLPBlock and ScoreNet: the network",
         """<p>A plain residual MLP. Given a point x (d numbers) and a step t:</p>
<div class="formula">z = Linear(x)  +  Linear(time embedding of t)
z = z + block(z)        × 4 blocks,  block = LayerNorm → act → Linear → act → Linear
output = Linear(act(LayerNorm(z)))      (d numbers)</div>
<p>For DDIM the output is the predicted <b>noise</b>; for flow matching it is a
<b>velocity</b>. Same network either way. The activation is LeakyReLU(0.2). Hidden width
<code>h</code> is 256 or 1024 (next piece).</p>
<p>The comment on line 143 matters: the attribute names (<code>inp</code>, <code>te</code>,
<code>tm</code>, <code>bl</code>, <code>out</code>) are the keys saved in checkpoint files, so
renaming them would make old checkpoints unloadable.</p>"""),
        ("def net_size", "net_size() / net_arch(): how big the network is",
         """<p><b>small</b> (width 256) for d ≤ 64, <b>large</b> (width 1024) above. The reason, from
the docstring: the R99 shell gets relatively thinner as d grows (about 1/√d), while a fixed-size
network's error does not shrink, so high-d models need more capacity to keep samples inside the
balls.</p>"""),
        ("def step_generator", "step_generator(): a private random stream per model",
         """<p>Every random number used while training one model (batch choice, time steps, noise)
comes from this generator and never from PyTorch's global one. Consequence: a seed trains to the
<b>bit-identical</b> model whether it is trained alone or together with other seeds in one CUDA
graph. The unit tests check exactly that.</p>"""),
        ("def learned_closure", "learned_closure(): one DDIM training step",
         """<p>Returns a fresh network and a function <code>step()</code> that computes one training
loss. A function that remembers its surroundings like this is called a <b>closure</b>; it lets the
optimiser code (next section) train any process without knowing its details.</p>
<p>One step, in words:</p>
<ol><li>take a batch of clean points x₀;</li>
<li>pick a random step i for each;</li>
<li>add the right amount of noise: x_i = √ᾱ_i·x₀ + √(1−ᾱ_i)·noise;</li>
<li>ask the network to predict that noise; the loss is the mean squared error.</li></ol>
<p>That is the standard "ε-prediction" diffusion loss.</p>"""),
        ("def _ema_decay", "The training recipe: EMA warm-up and cosine learning rate",
         """<p>Two small helpers that give a number per step:</p>
<ul>
<li><code>_ema_decay</code>: how strongly the <b>EMA</b> (a running average of the weights) holds on
to its past. It ramps up from about 0 over the first 1000 steps to 0.999.</li>
<li><code>_cosine_lr</code>: the learning rate, falling from <code>lr</code> = 10⁻³ to
<code>lr_min</code> = 10⁻⁵ along half a cosine. It reproduces PyTorch's
<code>CosineAnnealingLR</code> exactly, because the CUDA-graph path below has to compute the whole
table in advance.</li>
</ul>"""),
        ("def run_optimizers", "run_optimizers(): train several models at once",
         """<p>The recipe: <b>Adam</b>, <b>cosine learning-rate decay</b>, <b>gradient clipping</b> at
norm 1, and an <b>EMA</b> of the weights that replaces the weights at the end (averaged weights
usually sample better). Each piece is switched off by setting it to null.</p>
<p>Two implementations of the same maths:</p>
<ul>
<li><b>eager</b>: an ordinary Python loop. The reference, used on CPU.</li>
<li><b>CUDA graph</b>: on a GPU. These networks are so small that a Python loop spends most of its
time just <i>launching</i> GPU work. Recording one training step of all models once and replaying the
recording is about 15× faster.</li>
</ul>
<p><code>branch_streams=False</code> exists because several multi-stream graphs from different
processes on one GPU crashed with the driver in use ("illegal memory access").</p>"""),
        ("def _run_optimizer_eager", "_run_optimizer_eager(): the plain loop",
         """<p>Read this one to understand the recipe; the graph version does the same thing.
Per step: compute the loss, back-propagate, clip the gradient, take an Adam step, advance the
learning-rate schedule, update the EMA copy (<code>ema = decay·ema + (1−decay)·weights</code>). At
the end, load the EMA weights into the model.</p>"""),
        ("class _GraphBranch", "_GraphBranch: one model inside the recording",
         """<p>A CUDA graph replays the <i>same</i> GPU operations every time, so nothing may change on
the Python side between replays. The tricks used:</p>
<ul>
<li>The learning rate and EMA decay of every step are precomputed into tables <b>on the GPU</b>,
and a step counter that lives on the GPU picks the current entry.</li>
<li>Adam is created with <code>capturable=True</code> (its internal step count also stays on the
GPU).</li>
<li>Gradients are zeroed, not freed, because the recording owns those buffers.</li>
<li>The EMA update uses the fast <code>_foreach</code> operations on all weights at once.</li>
</ul>"""),
        ("def _run_optimizers_graph", "_run_optimizers_graph(): record once, replay many times",
         """<ol>
<li>Build the learning-rate and EMA tables and a GPU step counter.</li>
<li>Give each model its own CUDA <b>stream</b> so their steps run side by side.</li>
<li>Run 3 real steps first (they create the optimiser's memory).</li>
<li>Register each model's private random generator with the graph, so that every replay draws
<i>new</i> random numbers, then record one round.</li>
<li>Replay it for the remaining steps and hand back the models with EMA weights.</li>
</ol>"""),
        ("def true_score", "true_score(): the exact score of a Gaussian mixture",
         """<p>The formula that makes this project possible. For a mixture with centres m<sub>k</sub>,
common variance v and weights π<sub>k</sub>:</p>
<div class="formula">score(x) = Σ_k r_k(x) · (m_k − x) / v
r_k(x)   = softmax_k( log π_k − |x − m_k|² / 2v )</div>
<p>r<sub>k</sub> is the <b>responsibility</b>: how much x "belongs" to mode k. So the score is an
arrow from x towards a weighted average of the centres. Points midway between two modes are pulled
both ways, and that is where hallucinations are born.</p>
<p>After noising to level ᾱ the mixture is still a Gaussian mixture, with centres √ᾱ·μ<sub>k</sub>
and variance ᾱσ² + (1−ᾱ). So the exact score is known at <b>every</b> step. <code>log_weights</code>
just prepares log π for the unequal-weights variant.</p>"""),
        ("def _exact_eps", "_exact_eps(): the score, written as a noise prediction",
         """<p>The learned network predicts noise ε, not the score. The two are related by
<code>ε = −√(1−ᾱ) · score</code>, so this converts the exact score into exactly what a perfect
network would output. Now the exact and learned samplers can share the same update rule.</p>"""),
        ("def _ddim_step", "_ddim_step(): one deterministic DDIM step",
         """<p>From noise level ᾱ to level ᾱ′, given a noise estimate ε:</p>
<div class="formula">x₀_guess = (x − √(1−ᾱ)·ε) / √ᾱ          ← "what the clean point probably is"
x′       = √ᾱ′ · x₀_guess + √(1−ᾱ′) · ε    ← re-noise it to the next level</div>
<p>No fresh randomness enters, which is why every seed has one fixed fate. Going to a lower noise
level moves towards data; going to a higher one moves back towards the seed.</p>"""),
        ("def _ddim_transport", "_ddim_transport(): many steps with the exact score",
         """<p>Moves points through a list of noise levels using <code>_exact_eps</code> and
<code>_ddim_step</code>. With <code>order="heun"</code> (the default) each step is taken twice: once
to guess the end point, then again with the noise estimate averaged between start and guessed end.
That is second-order accurate, and precise enough that data → seed → data returns every point to the
same label.</p>
<p><code>inplace=True</code> overwrites the input tensor instead of copying it. At d = 512 an anchor
set can be ~15 GB, so a copy may not fit.</p>"""),
        ("def backtrack_true", "backtrack_true(): data → seed",
         """<p>Walks the noise levels upwards (ᾱ<sub>0</sub> → ᾱ<sub>T−1</sub>): a point in data space
becomes the seed that the exact sampler would have turned into it. <b>Step 2 of the atlas.</b></p>"""),
        ("def forward_true", "forward_true(): seed → data",
         """<p>The same walk downwards: the exact sampler proper. It is the inverse of
<code>backtrack_true</code>. Used to label calibration seeds, in the start-page figure and in
rank_probe.</p>"""),
        ("def ball_anchors", "ball_anchors(): step 1 of the atlas",
         """<figure>{{ANCHORS}}<figcaption>Schematic in 2-D (the real ones live in up to 512 dimensions).
Left: this function. Right: the next one.</figcaption></figure>
<p>For every mode k:</p>
<ul>
<li><b>b points uniformly inside the R99 ball</b>, labelled k. "Uniform in a ball" needs radius
R99·u<sup>1/d</sup> with u uniform in [0, 1]: in d dimensions most of a ball's volume is near its
edge, and the power 1/d puts the points there accordingly.</li>
<li><b>b × shell_frac points in the band</b> from R99 to R99 + 2σ, labelled −1. These represent
"just missed the mode": the hallucination class.</li>
</ul>
<p>The labels come from <a href="#label_fate">label_fate</a> itself, so they follow exactly the
rule the ground truth uses. <code>torch.addcmul(..., out=...)</code> writes straight into one
pre-allocated tensor to avoid temporary copies.</p>"""),
        ("def altered_knn_anchors", "altered_knn_anchors(): rings instead of a filled ball",
         """<p>The anchors for the altered kNN predictor. Per mode, 5 concentric <b>spheres</b> at
0.3, 0.6, 0.9, 1.2 and 1.5 × R99. <b>All</b> of them are labelled with the mode, even the two rings
outside R99; there is no hallucination class here. Instead each ring has a <b>weight</b> that falls
with its radius (linearly from 0.84 to 0.2, or a Gaussian). altered_knn later calls a seed a
hallucination when the vote of its neighbours is <i>unsure</i>, not because a neighbour was
labelled −1.</p>"""),
    ],
}

# =============================================================================================
# processes/
# =============================================================================================
PROC_FILES = [
    {"path": "code/processes/base.py",
     "intro_title": "One interface, two processes",
     "intro": """<p>train.py and evaluate.py never ask "is this DDIM or flow?". They call five methods,
and each process implements them:</p>
<div class="tablewrap"><table>
<tr><th>method</th><th>what it does</th></tr>
<tr><td><code>train_closure</code></td><td>a fresh network and its one-step training loss</td></tr>
<tr><td><code>sample</code></td><td>seeds → endpoints with the <b>learned</b> network</td></tr>
<tr><td><code>true_field_forward</code></td><td>seeds → endpoints with the <b>exact</b> field</td></tr>
<tr><td><code>true_field_backtrack</code></td><td>endpoints → seeds with the exact field</td></tr>
<tr><td><code>seeds</code></td><td>reproducible N(0, I) starting points</td></tr>
</table></div>
<p>Adding a third process means writing one more subclass and registering it in
<code>factory.py</code>.</p>""",
     "notes": [
         ("class Process", "Process: the shared part",
          """<ul>
<li>The constructor stores the mode centres, the variance σ², the number of steps T and the
mixing weights (plus their logarithms, which the exact field needs).</li>
<li><code>seeds(N, d, seed)</code>: N standard-normal points from their own generator, so seed 1
always gives the same points.</li>
<li><code>arch</code>, <code>n_train</code>, <code>optim_kwargs</code>: read the network size,
training-set size and recipe from the configuration.</li>
<li><code>label(model, X0, R99)</code>: seeds → fates. Pass <code>model=None</code> to use the exact
field instead of the network.</li>
</ul>"""),
         ("@abstractmethod", "The methods each process must write",
          "<p>Python refuses to create a process class that is missing any of these.</p>"),
     ]},
    {"path": "code/processes/ddim.py",
     "intro_title": "DDIM in one sentence",
     "intro": """<p>Diffusion with a deterministic sampler: the network learns to predict the noise
that was added, and sampling removes it one step at a time with
<a href="core.html#_ddim_step">core._ddim_step</a>. Step index T−1 is pure noise, index 0 is data.</p>""",
     "notes": [
         ("class DDIMProcess", "DDIMProcess",
          """<p>Builds the noise schedule ᾱ for its own T. That is T_train for the learned sampler
and T_true for the exact one, so the two schedules are different arrays (see the note on
<a href="core.html#make_schedule">make_schedule</a>). <code>true_order</code> chooses Heun (default)
or Euler for the exact field.</p>
<p><code>train_closure</code> just calls <a href="core.html#learned_closure">core.learned_closure</a>.</p>"""),
         ("def sample", "sample(): run the learned sampler",
          """<p>From i = T−1 down to 1: ask the network for the noise at step i, take one DDIM step.
Done in chunks of 50 000 seeds to bound memory. <code>model.eval()</code> and
<code>@torch.no_grad()</code> make it pure inference.</p>"""),
         ("def true_field_forward", "The exact field, both directions",
          """<p>Thin wrappers around <a href="core.html#forward_true">core.forward_true</a> and
<a href="core.html#backtrack_true">core.backtrack_true</a>, passing this process's schedule and
weights.</p>"""),
     ]},
    {"path": "code/processes/flow.py",
     "intro_title": "Flow matching in one paragraph",
     "intro": """<p>Instead of adding and removing noise, flow matching learns a <b>velocity field</b>:
at time t, which way and how fast should a point move so that, starting from noise at t = 0, it
arrives at data at t = 1? With the "optimal transport" path used here, each training pair (noise
x₀, data x₁) is joined by a straight line:</p>
<div class="formula">position  ψ_t = (1 − (1 − σ_min)·t)·x₀ + t·x₁
velocity  u   = x₁ − (1 − σ_min)·x₀</div>
<p>Sampling integrates the learned velocity from t = 0 to t = 1 with an ODE solver.</p>""",
     "notes": [
         ("class FlowOTProcess", "FlowOTProcess and its training step",
          """<p>The time grid is <code>T</code> evenly spaced points from 0 to 1. So unlike DDIM, a
different T only changes the step size, never the start or end.</p>
<p>One training step: take data x₁ and noise x₀, a random time t, the point ψ<sub>t</sub> on the
line between them, and train the network to output the line's velocity. The network's time input is
t rounded to the nearest grid index, so the same <code>ScoreNet</code> as DDIM can be used.</p>"""),
         ("def _step", "_step(): one ODE step",
          """<p>Four standard solvers: <b>Euler</b> (one velocity per step), <b>midpoint</b> and
<b>Heun</b> (two), <b>RK4</b> (four). More evaluations per step means more accuracy for the same
number of steps. The learned sampler uses Euler by default and the exact field uses Heun.</p>"""),
         ("def _integrate", "_integrate(): run the solver over the whole grid",
          """<p>Forwards (t: 0 → 1, seed → data) or backwards (t: 1 → 0, data → seed) with a negative
step, in chunks of 50 000 points.</p>"""),
         ("def sample", "sample(): the learned velocity, integrated",
          "<p>Wraps the network as a function f(x, t) and integrates it.</p>"),
         ("def _true_velocity", "_true_velocity(): the exact velocity of a Gaussian mixture",
          """<p>The flow version of <a href="core.html#true_score">true_score</a>. At time t, a point
from mode k is distributed as N(t·μ<sub>k</sub>, var<sub>t</sub>·I) with
var<sub>t</sub> = t²σ² + s<sub>t</sub>². Given a position x, standard Gaussian conditioning gives
the expected data point, and the velocity follows from it:</p>
<div class="formula">E[x₁ | x, mode k] = μ_k + gain · (x − t·μ_k),   gain = t·σ² / var_t
velocity(x, t)    = (Σ_k r_k · E[x₁ | x, k] − (1 − σ_min)·x) / s_t</div>
<p>with responsibilities r<sub>k</sub> as before. The docstring explains why the within-mode
variance σ² is kept: without it the field becomes extremely steep near t = 1 and the backward
integration cannot undo the forward one.</p>"""),
         ("def true_field_forward", "The exact field, both directions",
          "<p>Integrate the exact velocity forwards or backwards with <code>true_solver</code> (Heun).</p>"),
     ]},
    {"path": "code/processes/factory.py",
     "intro_title": "Choosing a process by name",
     "intro": """<p><code>make_process("ddim", ...)</code> or <code>make_process("flow", ...)</code>;
the name comes from <code>process=...</code> in the configuration. An unknown name fails with the
list of valid ones.</p>"""},
]

# =============================================================================================
# train.py
# =============================================================================================
TRAIN_FILE = {
    "path": "code/train.py",
    "intro_title": "Stage 1 in a nutshell",
    "intro": """<p>For every (d, K) in the sweep and every repeat seed: place the modes, train a network
on samples of the mixture, check that it rarely hallucinates, save it, and record where it sends
the evaluation seeds (the <b>ground truth</b> used by stage 2).</p>
<p>A checkpoint that already exists (with the right T) is reused, so the stage can be re-run
safely at any time.</p>
<div class="formula">checkpoints/&lt;process&gt;/&lt;variant&gt;/
    checkpoints/model_d&lt;d&gt;_K&lt;K&gt;_s&lt;seed&gt;.pt   the trained network + everything about its cell
    gt_cache/d&lt;d&gt;_K&lt;K&gt;_s&lt;seed&gt;.pt           fates of the evaluation seeds under it
    manifest.json                               one summary line per model</div>""",
    "notes": [
        ("def variant_of", "Paths: where everything is stored",
         """<p>Small helpers shared by the other stages so that every file name is built in exactly
one place:</p>
<ul>
<li><code>variant_of</code>: "weighted" or "unweighted".</li>
<li><code>sweep_dir</code>: <code>output/&lt;run_id&gt;/&lt;process&gt;/&lt;variant&gt;/T&lt;T_true&gt;</code>,
where results go. <code>resolve_sweep_dir</code> returns it only if it already has results.</li>
<li><code>ckpt_path</code>, <code>gt_cache_path</code>: the two files per trained model.</li>
<li><code>n_eval_for</code>: 50 000 × K evaluation seeds, so every mode gets about the same number
whatever K is (K = 2 → 100 000, K = 16 → 800 000).</li>
</ul>"""),
        ("def _save_atomic", "_save_atomic(): never half a file",
         """<p>Saves to a temporary name first, then renames. A rename is instantaneous, so if the job
is killed mid-save the old file (or no file) remains, never a corrupt half-written one that a
later resume would trust.</p>"""),
        ("def save_gt_cache", "The ground-truth cache",
         """<p>Stores the fate of each evaluation seed plus what it was computed with.
<code>load_gt_cache</code> only accepts a cache made for the same number of seeds and the same
repeat seed; anything else (or an unreadable file) counts as missing and is recomputed.</p>"""),
        ("def _git_commit", "_git_commit(): which code made this model",
         """<p>The current commit hash, with "-dirty" if there are uncommitted changes. Saved inside
every checkpoint so you can always tell which version of the code trained it.</p>"""),
        ("class ModePlacementError", "Geometry of one cell",
         """<p><code>_cell_geometry</code> sets up one repeat: the sphere radius is
<code>data.radius × √(d/2)</code> (2.0 at d = 2, 11.3 at d = 64, 32 at d = 512) so the modes stay
well apart as d grows; then the centres, R99 and the mixing weights. If the modes cannot be placed
it raises <code>ModePlacementError</code>, and the cell is skipped instead of crashing the sweep.</p>"""),
        ("def _save_cell", "_save_cell(): write the checkpoint and the ground truth",
         """<p>A checkpoint holds far more than weights: the mode centres and weights, R99, σ, T, the
hallucination rate measured at the end of training, whether it converged, how many steps and
attempts it took, the network size, the <b>entire configuration</b> and the git commit. Evaluation
needs nothing else to rebuild the cell.</p>
<p>Then it runs the new model on the evaluation seeds (repeat seed + 1), labels the endpoints with
<a href="core.html#label_fate">label_fate</a> and caches them. If that fails (for example out of
memory), evaluate.py simply recomputes it later.</p>"""),
        ("def _ckpt_T", None, "<p>Reads the step count stored in an existing checkpoint (None if it cannot be read).</p>"),
        ("def train_cell", "train_cell(): train every missing model of one (d, K)",
         """<ol>
<li><b>Skip what exists.</b> A checkpoint with the right T is reused. A checkpoint made with another
T is retrained.</li>
<li><b>Set up geometry</b> for each remaining repeat seed.</li>
<li><b>How long to train.</b> The number of steps grows with d and K:
<div class="formula">steps = base_steps × (1 + d/16) × (1 + K/16)
  d=2,  K=2   →    37 968        d=64,  K=16  →   300 000
  d=16, K=8   →    90 000        d=512, K=16  → 1 980 000</div></li>
<li><b>Train all pending seeds together</b> (one CUDA graph; see
<a href="core.html#run_optimizers">run_optimizers</a>).</li>
<li><b>Probe.</b> Sample 10 000 fresh seeds (repeat seed + 7) and measure the hallucination rate.
Above <code>hall_target</code> = 1.5%? Train that seed <i>again from scratch</i> with 1.7× more
steps. At most <code>max_attempts</code> = 2 tries; a model that still misses is saved but flagged
<code>not_converged</code>.</li>
<li><b>Save</b> every model with <code>_save_cell</code> and return one summary per seed.</li>
</ol>"""),
        ("def run", "run(): the whole stage",
         """<p>Loops over every d and K of the sweep, prints one line per model, then merges the
summaries into <code>manifest.json</code> under a file lock (parallel jobs write the same file). It
ends with a warning if any model stayed above the hallucination target, because then that cell's
results reflect a badly trained model rather than the geometry being studied.</p>"""),
    ],
}

# =============================================================================================
# evaluate.py
# =============================================================================================
EVAL_FILE = {
    "path": "code/evaluate.py",
    "intro_title": "Stage 2 in a nutshell",
    "intro": """<p>The experiment itself. For every (d, K, repeat seed) cell and every anchor budget b:</p>
<ol>
<li><b>Plant anchors</b> around the modes in data space
(<a href="core.html#ball_anchors">core.ball_anchors</a>).</li>
<li><b>Backtrack</b> them to seed space with the exact field.</li>
<li><b>Fit</b> each predictor on the (seed, fate) pairs (<a href="fate.html">fate.py</a>).</li>
<li><b>Ground truth</b>: fates of 50 000·K fresh seeds under the learned sampler (cached by train.py).</li>
<li><b>Score</b> each predictor against it.</li>
</ol>
<p>Results are saved per cell, so an interrupted or partial evaluation resumes where it stopped,
and a new budget or predictor only computes its own rows.</p>""",
    "notes": [
        ("def load_ckpt", "load_ckpt(): rebuild a trained network",
         """<p>Reads a checkpoint, rebuilds a <code>ScoreNet</code> of the stored size, loads the
weights and switches it to inference mode. Returns the model and the whole checkpoint dict (centres,
R99, T…).</p>"""),
        ("def budgets", "Reading the configuration",
         """<ul>
<li><code>budgets</code>: the anchor budgets as a list of ints (accepts a single number too).</li>
<li><code>_specs</code>: each predictor's final settings: the shared defaults from
<code>conf/classifier/base.yaml</code>, overridden by its own file.</li>
<li><code>_primary</code>: the predictor summarised on the console (the first one unless set).</li>
</ul>"""),
        ("def preflight", "preflight(): fail in seconds, not hours",
         """<p>Estimates the GPU memory the biggest budget will need (all anchors, the ring anchors, the
evaluation seeds, at 4 bytes per number, plus ~3 GB working space) and stops immediately if it is
above 92% of the free memory, with advice on what to shrink. Without this, a d = 512 job could run
for hours and then die of an out-of-memory error.</p>"""),
        ("def cell_path", "Cell files: the permanent record",
         """<p>Each cell's results live in <code>cells/d&lt;d&gt;_K&lt;K&gt;_s&lt;seed&gt;.json</code>
as a list of rows, one per (budget, predictor). Saved atomically (temporary file + rename).
<code>_key</code> identifies a row; <code>row_order</code> sorts rows in one fixed order so that
files built in any order end up identical.</p>"""),
        ("def _ground_truth", "_ground_truth(): what the learned model really does",
         """<p>Makes the evaluation seeds (repeat seed + 1, the same as train.py used) and takes their
fates from train.py's cache, or computes and caches them if the cache is missing.</p>"""),
        ("def eval_one", "eval_one(): one cell, start to finish",
         """<p><b>The heart of the project.</b></p>
<ol>
<li><b>What is missing?</b> Compare the wanted (budget, predictor) rows with the cell file. Nothing
missing → return straight away. <code>eval.force=true</code> recomputes the requested rows.</li>
<li><b>Load</b> the trained model and build <i>two</i> processes on the same mixture:
<code>proc</code> with the model's T (the learned sampler) and <code>proc_true</code> with T_true
(the exact field).</li>
<li><b>Ground truth</b>, then free the network (only its fates are needed from here on).</li>
<li><b>Calibration seeds</b>: 5 000 seeds (repeat seed + 2) labelled by the <i>exact</i> field.
altered_knn uses them to choose its hallucination threshold.</li>
<li>For each budget:
<ul>
<li>ball + shell anchors → backtrack to seeds (step 1–2), in place to save memory;</li>
<li>ring anchors too, if altered_knn is among the predictors;</li>
<li>for each predictor: fit (step 3), calibrate if needed, predict the evaluation seeds' fates,
compute the scores (steps 4–5) and store a row;</li>
<li>free the anchors before the next, bigger budget.</li>
</ul></li>
<li>Merge new rows with the cached ones, sort, save the cell file.</li>
</ol>
<p>A row records the cell, the budget, the number of anchors, the predictor, the learned sampler's
hallucination rate (<code>hall_gt</code>), the counts, the time taken and every score.</p>"""),
        ("AGG_KEYS", "aggregate(): mean ± std over repeat seeds",
         """<p>Groups rows by (d, K, budget, predictor) and replaces each score by its mean over the
repeat seeds, plus a <code>_std</code> companion (population standard deviation). Missing values
(NaN) are ignored rather than poisoning the average.</p>"""),
        ("def _console", None, """<p>One summary line per (d, K) on the console: the learned sampler's
hallucination rate and the primary predictor's best budget.</p>"""),
        ("def run", "run(): the whole stage",
         """<p>For each (d, K): preflight check, evaluate every repeat seed, print the summary. A cell
without a checkpoint is reported and skipped.</p>
<p>With <code>eval.part=true</code> (what scripts/main.sh uses, one job per d) it stops after
writing cell files; a single later <code>merge</code> builds the results. Otherwise it merges
straight away.</p>"""),
        ("def _write_results", "_write_results(): the three results files",
         """<ul>
<li><code>results.json</code>: everything: the sweep that is covered, every per-seed row and the
aggregate. This is the only file visualize.py reads.</li>
<li><code>results.csv</code>: the aggregate table (mean and std of each score).</li>
<li><code>results_per_seed.csv</code>: one line per row, with the mixing weights.</li>
</ul>"""),
        ("def merge", "merge(): rebuild the results from every cell file",
         """<p>Reads <i>all</i> cell files of this (process, variant, T_true), whichever invocation or
machine wrote them, and rewrites the results files. Protected by a lock file because parallel jobs
may call it at the same moment.</p>"""),
    ],
}

# =============================================================================================
# fate.py
# =============================================================================================
FATE_FILE = {
    "path": "code/fate.py",
    "intro_title": "The predictors",
    "intro": """<p>Four ways to guess a seed's fate from its coordinates. All of them output K+1 scores
per seed: column 0 for "hallucination", columns 1…K for the modes.</p>
<div class="tablewrap"><table>
<tr><th>name</th><th>kind</th><th>idea</th></tr>
<tr><td><b>knn</b></td><td>non-parametric</td><td>look at the 10 nearest backtracked anchors and take a vote</td></tr>
<tr><td><b>altered_knn</b></td><td>non-parametric</td><td>weighted vote among mode anchors only; an <i>unsure</i> vote means hallucination</td></tr>
<tr><td><b>quadratic</b></td><td>trained</td><td>a linear classifier on x and all products x<sub>i</sub>x<sub>j</sub>: curved (quadratic) boundaries</td></tr>
<tr><td><b>polar3</b></td><td>trained</td><td>describes a seed by direction and radius; hallucination = close to the boundary between two modes</td></tr>
</table></div>""",
    "notes": [
        ("PARAMETRIC", None, "<p>Which predictors are trained networks and which just store the anchors.</p>"),
        ("def _chunk", "_chunk(): keep distance matrices small",
         """<p>Comparing each query seed with millions of anchors needs a big distance matrix. This
picks how many queries to handle at once so the matrix stays near 2·10⁸ numbers (~800 MB).</p>"""),
        ("class KNN", "KNN: the nearest-neighbour vote",
         """<p><code>fit</code> just stores the anchors and their labels (one-hot, hallucination in
column 0). <code>forward</code> finds each seed's k = 10 nearest anchors and returns the log of the
share of votes for each class. The class with the most votes wins.</p>"""),
        ("class AlteredKNN", "AlteredKNN: hallucination = low confidence",
         """<p>Trained only on <a href="core.html#altered_knn_anchors">ring anchors</a>, which are all
labelled with a mode. For a seed:</p>
<ol>
<li>take the 10 nearest anchors; each votes for its mode with its ring weight;</li>
<li>sharpen the vote shares with a softmax at temperature 0.1;</li>
<li><b>confidence</b> = 1 − entropy / log K (1 = all neighbours agree, 0 = evenly split), or
simply the largest share;</li>
<li>confidence below the threshold → hallucination.</li>
</ol>
<p><code>calibrate</code> picks the threshold automatically: it tries 100 cut-offs (the 0.5% to 50%
quantiles of the confidences on the calibration seeds) and keeps the one with the best
hallucination F1 against the exact field's labels. <code>forward</code> turns the decision into
a hard ±30 score in column 0 so that it always wins or always loses.</p>"""),
        ("class FateNet", "FateNet: the two trained predictors",
         """<p><b>quadratic</b>: the input is x plus every product x<sub>i</sub>·x<sub>j</sub>
(i ≤ j), fed to one linear layer. That can draw any quadratic boundary (ellipsoids, hyperboloids).
There are d(d+1)/2 products: 528 at d = 32, 131 328 at d = 512.</p>
<p><b>polar</b>: a seed x ~ N(0, I) has length close to √d, so it is described by its
<b>direction</b> x/|x| and how far its length is from √d. A projection to 256 features and their
powers 1…3 (a degree-3 polynomial) give the K mode scores. The hallucination score is</p>
<div class="formula">c + a · (|x| − √d) − β · (best mode score − second-best mode score)</div>
<p>In words: a seed is likely to hallucinate when the model can hardly decide between its two best
modes, i.e. near a boundary between their regions. The radius term lets that depend on how far out
the seed is. c, a and β are learned.</p>"""),
        ("def train_fate_classifier", "train_fate_classifier(): fit one predictor",
         """<p>kNN-type predictors are "fitted" by storing the anchors. The trained ones use AdamW with a
one-cycle learning rate (warm up, then decay), cross-entropy loss, at least 30 epochs and at least
<code>min_steps</code> = 3000 gradient steps. Labels are shifted by one so that −1 → class 0.</p>"""),
        ("def train_ensemble", "train_ensemble(): three models are steadier than one",
         """<p>Trains <code>ensemble</code> = 3 copies of a trained predictor with seeds 0, 1, 2 (just one
for kNN-type predictors, which have no randomness). Their predictions are combined below.</p>"""),
        ("def predict_fate", "predict_fate(): scores → a fate",
         """<p>Adds the ensemble's log-probabilities. The seed is called a hallucination if the
hallucination score beats the best mode score, otherwise it gets the best mode. For quadratic the
chunk is shrunk so the d(d+1)/2 product features stay under ~2 GB.</p>"""),
        ("METRICS = ", "fate_metrics(): how the atlas is graded",
         """<div class="tablewrap"><table>
<tr><th>score</th><th>question it answers</th></tr>
<tr><td><code>full_acc</code></td><td>share of all seeds whose fate was guessed exactly</td></tr>
<tr><td><code>mode_acc</code></td><td>among seeds that truly go to a mode: share given the right mode</td></tr>
<tr><td><code>mode_f1</code></td><td>F1 per mode, averaged over modes (so small modes count equally)</td></tr>
<tr><td><code>hall_prec</code></td><td>of the seeds <i>called</i> hallucinations, the share that really are</td></tr>
<tr><td><code>hall_rec</code></td><td>of the seeds that <i>really</i> hallucinate, the share that were caught</td></tr>
<tr><td><code>hall_f1</code></td><td>the balance of the two: 2·P·R / (P + R)</td></tr>
<tr><td><code>balanced</code></td><td>average of mode_acc and hall_f1</td></tr>
</table></div>
<div class="callout"><p><b>Example.</b> 1000 seeds, 20 of which really hallucinate. A predictor
flags 30 seeds, 15 of them correctly. Precision = 15/30 = 0.50, recall = 15/20 = 0.75,
F1 = 0.60.</p>
<p>When real hallucinations are very rare (say 0.2% of seeds), even a good detector gets a low
precision, because its false alarms outnumber the few true cases. Read hall_f1 together with
<code>hall_gt</code>, the true hallucination rate stored in each row.</p></div>"""),
    ],
}

# =============================================================================================
# visualize.py
# =============================================================================================
VIZ_FILE = {
    "path": "code/visualize.py",
    "intro_title": "Stage 3 in a nutshell",
    "intro": """<p>Reads <code>results.json</code> and nothing else, so figures can be restyled and
redrawn in seconds (<code>stages=[visualize]</code>) without recomputing anything. All numbers are
shown in percent as mean ± std over the repeat seeds. <code>matplotlib.use("Agg")</code> draws to
files without needing a screen (important on servers).</p>""",
    "notes": [
        ("PANELS", None, "<p>The three scores shown side by side in every heatmap figure.</p>"),
        ("def best_rows", "best_rows(): one number per (d, K)",
         """<p>A cell may have been evaluated at several anchor budgets. For the summary figures each
predictor gets its best budget, judged by mean full accuracy.</p>"""),
        ("def plot_model", "plot_model(): the heatmaps",
         """<p>Three panels (full accuracy, mode F1, hallucination F1). Dimension d runs left to right,
the number of modes K top to bottom, colour = mean score from 0 to 1, and each square is labelled
"mean±std" in percent. Empty cells stay blank; the text turns white on dark squares.</p>"""),
        ("def table_rows", None, "<p>The table as (K, d, [(mean, std) per predictor]) rows, at each predictor's best budget.</p>"),
        ("def write_tex", "write_tex(): a table ready for the paper",
         """<p>A LaTeX table using the <code>booktabs</code> package, one block of rows per K separated
by a rule, "--" where there is no result.</p>"""),
        ("def write_png", "write_png(): the same table as an image", "<p>Bold header, striped rows; for a quick look without LaTeX.</p>"),
        ("def run", "run(): draw everything",
         """<p>Finds this run's <code>results.json</code> (never another run's), then writes the table
(tex + png), one heatmap figure per predictor at its best budget, and one per predictor for each
budget in <code>anchors_&lt;b&gt;/</code>. Each figure is attempted separately, so one failing
figure is reported without stopping the rest.</p>"""),
    ],
}

# =============================================================================================
# combine.py
# =============================================================================================
COMBINE_FILE = {
    "path": "code/combine.py",
    "intro_title": "When a run is split across machines",
    "intro": """<p>For example d ≤ 256 on the lab server and d = 512 on a cluster, both with the same
<code>run_id</code> and settings. Copy the other machine's <code>checkpoints/</code> and
<code>output/</code> into one folder, then:</p>
<div class="formula">python code/combine.py /tmp/other --dry-run    # report only
python code/combine.py /tmp/other              # merge, then rebuild results and figures</div>
<p>The result is the same as if one machine had done everything. The one rule it never breaks:
<b>results computed from one model are never mixed with a different model of the same cell.</b></p>""",
    "notes": [
        ("REBUILT", None, "<p>Files that are always rebuilt from the cell files, so the other side's copies are never taken.</p>"),
        ("def _copy", "Safe file writing", "<p><code>_copy</code> and <code>_write_json</code> write to a temporary file and rename it (atomic); in dry-run mode they do nothing.</p>"),
        ("def _relocate", None, "<p>The other machine's manifest stores absolute paths such as <code>/home/them/iclr_code/checkpoints/…</code>; this rewrites them to point at this machine's <code>checkpoints/</code>.</p>"),
        ("def _same_model", "_same_model(): are these the same network?",
         """<p>First a byte-for-byte comparison (fast). If the files differ, it loads both and compares
the weights tensor by tensor, because two saves of the same model can differ in metadata such as
the creation time.</p>"""),
        ("def check_settings", "Step 1: same settings?",
         """<p>Compares the two <code>run.json</code> files with the same check every invocation does
(<a href="main.html#settings_diff">runstate.settings_diff</a>). Different settings mean different
experiments, so it refuses unless you pass <code>--force</code>. New processes or predictors from
the other side are added.</p>"""),
        ("def _conflicts", "Step 2: the same cell trained twice?",
         """<p>Lists every model that exists on both sides with <i>different</i> weights. For those cells
this side's model is kept, and nothing the other side computed from its own model (its ground-truth
cache, its result rows) is copied.</p>"""),
        ("def combine", "combine(): step 3, the files",
         """<p>Walks the other side's <code>checkpoints/</code> and run folder and, file by file:</p>
<ul>
<li>skips results files and figures (rebuilt later), <code>run.json</code> and
<code>history.log</code> (merged separately) and anything belonging to a conflicting cell;</li>
<li><code>manifest.json</code>: union of both, this side winning ties;</li>
<li>a file this side lacks: copied;</li>
<li>identical files, or a model with the same weights: nothing to do;</li>
<li>a cell file on both sides: rows this side lacks are added;</li>
<li>anything else that differs: this side's copy is kept, and reported.</li>
</ul>
<p>Finally the two history logs are merged and a "combine" line is added. It returns counts of what
happened.</p>"""),
        ("def rebuild", "rebuild(): step 4, results and figures",
         """<p>For every <code>&lt;process&gt;/&lt;variant&gt;/T&lt;T&gt;</code> folder with cell files,
it builds a configuration with Hydra's <code>compose</code> (the same way main.py would), then runs
<a href="evaluate.html#merge">evaluate.merge</a> and <a href="visualize.html#run">visualize.run</a>.
A figure failing does not stop the rebuild.</p>"""),
        ("def main", "main(): the command line",
         """<p>Options: <code>--run</code> (needed only if the source has several runs),
<code>--into</code> (this repository's root; default <code>$ATLAS_ROOT</code> or the current
folder), <code>--force</code>, <code>--dry-run</code>, <code>--no-rebuild</code>.</p>"""),
    ],
}

# =============================================================================================
# rank_probe.py
# =============================================================================================
RANK_FILE = {
    "path": "code/rank_probe.py",
    "intro_title": "A separate side experiment",
    "intro": """<p>This file uses nothing else from the repository. It copies the few functions it
needs, so it can be pasted into a notebook. The question: <b>do the seeds that hallucinate fill seed
space in every direction, or do they lie on a thin, low-dimensional set?</b></p>
<p>It collects about 100 000 hallucinating seeds with the <i>exact</i> DDIM sampler (no trained
network), then measures how many dimensions they really use, compared with ordinary Gaussian noise
of the same size. Edit the CONFIG block and run it. Only numpy and torch are needed; scipy is
optional.</p>""",
    "notes": [
        ("d          = 32", "CONFIG: the knobs",
         """<p>The same toy world as the main code (d = 32, K = 8, σ = 0.1, the same schedule) with
T = 200 exact Heun steps. It keeps drawing batches of 200 000 seeds until it has 100 000
hallucinating ones or has tried 40 million seeds.</p>"""),
        ("def chi2_ppf", "chi2_ppf(): R99 without scipy",
         """<p>Uses scipy when available; otherwise the Wilson–Hilferty approximation, which turns the
chi-square percentile into a normal one. Accurate enough for R99.</p>"""),
        ("def r99", "Copies of the core functions",
         """<p><code>r99</code>, <code>make_schedule</code>, <code>sample_modes</code>,
<code>true_score</code>, <code>_eps</code>, <code>_step</code>, <code>forward_map</code> and
<code>label_fate</code> are the same maths as in <a href="core.html">core.py</a> (equal weights
only), copied so the file stands alone.</p>"""),
        ("def rank_report", "rank_report(): how many dimensions does a point cloud use?",
         """<p>The singular values of the data matrix say how much the cloud spreads in each direction.
Several summaries:</p>
<ul>
<li><b>naive rank</b>: directions with any spread at all. With many noisy points it is always d, so
it cannot tell anything apart; reported only to show that.</li>
<li><b>effective rank</b>: exp(entropy of the normalised singular values). About d for round noise;
much smaller if a few directions dominate.</li>
<li><b>stable rank</b> and <b>participation ratio</b>: two more ways of counting the dominant directions.</li>
<li><b>n90 / n95 / n99</b>: how many principal directions hold 90 / 95 / 99% of the variance.</li>
</ul>
<p>The data is centred first (subtract the mean), which makes this a PCA.</p>"""),
        ("def mode_subspace_energy", "mode_subspace_energy(): lined up with the modes?",
         """<p>The K mode centres span at most K−1 directions. This measures what share of the
cloud's spread lies within those directions. Plain noise gives about (K−1)/d; a much higher value
means the hallucinating seeds are organised around the arrangement of the modes.</p>"""),
        ("def fmt", None, "<p>Formats one report as a console line.</p>"),
        ("# run", "Running the probe",
         """<ol>
<li>Set up the world: R99, mode centres, schedule.</li>
<li>Collect: draw seeds, run the exact sampler, keep the ones whose endpoint lands in no ball, and
print progress.</li>
<li>Build a Gaussian control set of the same size.</li>
<li>Report ranks for three clouds: hallucinating seeds (noise space), their endpoints (data space),
and the control; then the mode-subspace shares and the top singular values.</li>
<li>Print a plain-language reading ("READ:") and optionally save everything to an
<code>.npz</code> file.</li>
</ol>"""),
    ],
}

# =============================================================================================
# configuration + setup
# =============================================================================================
CONFIG_INTRO = """
<p>Nothing in the code has a hard-coded experiment setting: everything is read from
<code>conf/</code> through <b>Hydra</b>. <code>conf/config.yaml</code> picks one file from each
folder ("group"); any value can be changed on the command line without editing a file:</p>
<div class="formula">python code/main.py process=flow                       # pick another file of a group
python code/main.py sweep.d=[16,64] sweep.K=[4,8]       # change values
python code/main.py train.lr=5e-4 run_id=lr_test        # a result-changing setting → a new run_id</div>
<h3>The settings you are most likely to touch</h3>
<div class="tablewrap"><table>
<tr><th>setting</th><th>default</th><th>effect</th></tr>
<tr><td><code>sweep.d</code>, <code>sweep.K</code></td><td>[2], [2,4,8,16]</td><td>which cells to compute</td></tr>
<tr><td><code>sweep.anchors</code></td><td>[20000,50000,100000]</td><td>anchor budgets per mode</td></tr>
<tr><td><code>n_seeds</code></td><td>3</td><td>independent repeats (mean ± std)</td></tr>
<tr><td><code>process</code></td><td>ddim</td><td>ddim or flow</td></tr>
<tr><td><code>process.T_train</code> / <code>T_true</code></td><td>500 / 200</td><td>steps of the learned / exact sampler</td></tr>
<tr><td><code>data.weighted</code></td><td>false</td><td>unequal mixing weights</td></tr>
<tr><td><code>run_id</code></td><td>abc123</td><td>the name of the run folder</td></tr>
<tr><td><code>stages</code></td><td>[train, evaluate, visualize]</td><td>what to run</td></tr>
<tr><td><code>device</code></td><td>auto</td><td>cpu, cuda, or auto</td></tr>
</table></div>
<p>Settings that change results are locked per run (see <a href="main.html#check_and_record">runstate</a>):
change them together with a new <code>run_id</code>.</p>
"""

CONFIG_FILES = [
    {"path": "conf/config.yaml", "intro_title": "The top-level file",
     "intro": "<p>Everything else is pulled in from here.</p>",
     "notes": [
         ("defaults:", "defaults: one file per group",
          """<p>Each line <code>group: name</code> loads <code>conf/group/name.yaml</code> into
<code>cfg.group</code>. The <code>anchors</code> group uses <code>altered_knn.yaml</code>, which
itself builds on <code>anchors/base.yaml</code>.</p>"""),
         ("- classifier@classifier.models.knn", "The predictors to run",
          """<p>Each line puts one predictor file under <code>cfg.classifier.models.&lt;name&gt;</code>.
To skip a predictor, delete its line; to add one, add a line. (The group is called "classifier"
for historical reasons: these are the fate predictors.)</p>"""),
         ("- override hydra/job_logging", "Keep Hydra quiet",
          """<p>By default Hydra writes its own log files and a <code>.hydra/</code> folder into the
working directory. Switched off; main.py saves the configuration in the run folder instead.</p>"""),
         ("stages:", "Run-level settings",
          """<p>What to run, on which device, the base RNG seed, how many repeats, the run's name, and
whether a settings mismatch is an error (<code>strict_run</code>).</p>"""),
         ("paths:", "Where files go",
          """<p><code>root</code> is <code>$ATLAS_ROOT</code> if set (scripts/main.sh sets it),
otherwise the current directory. Models go to <code>checkpoints/</code>, results to
<code>output/</code>.</p>"""),
         ("hydra:", None, "<p>Stay in the current directory; Hydra must not change into a folder of its own.</p>"),
     ]},
    {"path": "conf/process/base.yaml", "intro_title": "Steps for both processes",
     "intro": """<p>T_train = 500 steps for the learned sampler; the comment records why 150 was
abandoned. T_true = 200 steps for the exact sampler that builds the atlas. For DDIM, see the
note on <a href="core.html#make_schedule">make_schedule</a> about T changing the total noise.</p>"""},
    {"path": "conf/process/ddim.yaml", "intro_title": "DDIM",
     "intro": "<p>The noise schedule's end points (β from 10⁻⁴ to 0.02) and the exact sampler's integrator (Heun).</p>"},
    {"path": "conf/process/flow.yaml", "intro_title": "Flow matching",
     "intro": """<p>σ<sub>min</sub>, the tiny spread left at t = 1 on the straight paths; the learned
sampler's solver (Euler) and the exact field's (Heun).</p>"""},
    {"path": "conf/data/base.yaml", "intro_title": "The data, shared part",
     "intro": """<p><code>mass_q</code> = 0.99 is where "99%" in R99 comes from: change it and the line
between a mode and a hallucination moves. <code>weighted</code> chooses the variant.</p>"""},
    {"path": "conf/data/gmm.yaml", "intro_title": "The Gaussian mixture",
     "intro": """<p>σ = 0.1 per mode; base sphere radius 2.0 (scaled by √(d/2) in the code); modes at
least 1.5 × 2 × R99 apart.</p>"""},
    {"path": "conf/sweep/base.yaml", "intro_title": "The grid",
     "intro": """<p>Which d, K and anchor budgets to compute. The defaults are a small grid (d = 2 only);
scripts/main.sh passes the full one.</p>"""},
    {"path": "conf/train/base.yaml", "intro_title": "Training the learned sampler",
     "intro": """<p>Every piece of the recipe in <a href="core.html#run_optimizers">run_optimizers</a>
and the retry rule in <a href="train.html#train_cell">train_cell</a>: the number of steps, the
learning rate and its floor, gradient clipping, EMA, batch size, network widths, training-set size,
the CUDA-graph switches, and the hallucination target with its retries.</p>"""},
    {"path": "conf/eval/base.yaml", "intro_title": "Evaluation",
     "intro": """<p>50 000 ground-truth seeds per mode; <code>part</code> (write cell files only) and
<code>force</code> (recompute rows) are per-invocation switches.</p>"""},
    {"path": "conf/anchors/base.yaml", "intro_title": "Anchors, shared part",
     "intro": """<p>The anchor RNG seed; the hallucination band (half as many points as the ball, 2σ
wide); 5 000 calibration seeds for altered_knn.</p>"""},
    {"path": "conf/anchors/altered_knn.yaml", "intro_title": "Ring anchors",
     "intro": """<p>5 rings out to 1.5 × R99, weights falling linearly to 0.2. See the
<a href="core.html#ball_anchors">picture</a>.</p>"""},
    {"path": "conf/classifier/base.yaml", "intro_title": "Predictor defaults",
     "intro": """<p>Every predictor starts from these values (<code>_shared</code>); its own file only
lists what differs. Some only matter to one kind of predictor, as the comments say.</p>"""},
    {"path": "conf/classifier/knn.yaml", "intro_title": "knn",
     "intro": "<p>Name and implementation only; everything else from the defaults.</p>"},
    {"path": "conf/classifier/altered_knn.yaml", "intro_title": "altered_knn",
     "intro": "<p>Name and implementation only; everything else from the defaults.</p>"},
    {"path": "conf/classifier/quadratic.yaml", "intro_title": "quadratic",
     "intro": "<p>Name and implementation only; everything else from the defaults.</p>"},
    {"path": "conf/classifier/polar3.yaml", "intro_title": "polar3",
     "intro": "<p>The polar predictor with a degree-3 polynomial. A <code>polar5.yaml</code> would only need a different name and degree.</p>"},
    {"path": "README.md", "intro_title": "The repository's own README",
     "intro": """<p>Setup, how to run, the output layout and the method notes, as the author wrote them.
The README's explanation of why 150 steps hallucinate ("small discretisation error") is plausible.
For DDIM, part of the effect may also be the schedule issue described at
<a href="core.html#make_schedule">make_schedule</a>.</p>"""},
    {"path": "requirements.txt", "intro_title": "Python packages",
     "intro": "<p>Tested with Python 3.12 and torch 2.5.1 (CUDA 12.1 wheels), per the README.</p>"},
    {"path": ".gitignore", "intro_title": "Not tracked by git",
     "intro": "<p>The virtual environment and Python's compiled caches.</p>"},
]

# =============================================================================================
# scripts
# =============================================================================================
SCRIPT_FILES = [
    {"path": "scripts/main.sh",
     "intro_title": "The whole sweep on every GPU",
     "intro": """<p>Runs the full grid in three phases, each spread over all GPUs as a queue of small
jobs: <b>train</b> (one job per cell), <b>evaluate</b> (one job per d), <b>merge + visualize</b>.
Every setting is an environment variable, and every job is an ordinary
<code>python code/main.py ...</code> call with overrides; <code>conf/</code> is never edited.</p>
<div class="formula">./scripts/main.sh                                       # the full grid
DIMS="512" KS="2 4 8 16" PROCESSES=ddim ./scripts/main.sh  # a slice
SKIP_TRAIN=true ./scripts/main.sh                       # evaluation only
GPUS="3" ./scripts/main.sh                              # only GPU 3</div>""",
     "notes": [
         ("set -o pipefail", "Shell setup",
          """<p><code>set -f</code> turns off filename expansion so that overrides like
<code>[2,4,8]</code> reach Python unchanged. The script moves to the repository root and exports
<code>ATLAS_ROOT</code> so every job writes to the same place.</p>"""),
         ("RUN=${RUN", "The settings (environment variables)",
          """<div class="tablewrap"><table>
<tr><th>variable</th><th>default</th><th>meaning</th></tr>
<tr><td>RUN</td><td>abc123</td><td>run name</td></tr>
<tr><td>PROCESSES</td><td>ddim flow</td><td>processes</td></tr>
<tr><td>DIMS / KS</td><td>2…256 / 2…32</td><td>the grid</td></tr>
<tr><td>TS</td><td>100 200 500</td><td>T_true values</td></tr>
<tr><td>ANCHORS</td><td>[20000,50000,100000]</td><td>budgets</td></tr>
<tr><td>NSEEDS</td><td>3</td><td>repeats</td></tr>
<tr><td>JOBS_PER_GPU / EVAL_JOBS_PER_GPU</td><td>1 / 3</td><td>concurrent jobs per GPU</td></tr>
<tr><td>WEIGHTED</td><td>false true</td><td>which variants</td></tr>
<tr><td>RETRIES</td><td>1</td><td>re-runs of a failed job</td></tr>
<tr><td>TRAIN_ONLY / SKIP_TRAIN</td><td>false</td><td>run part of the pipeline</td></tr>
</table></div>
<div class="callout warn"><p>The comment on <code>DATASET</code> says <code>gmm | mnist</code>, but
there is no <code>conf/data/mnist.yaml</code> in this repository any more, so only
<code>gmm</code> works.</p></div>"""),
         ("if [ -z \"${PY", "Finding Python", "<p>Uses <code>$PY</code> if given, else <code>python3</code>, else <code>python</code>.</p>"),
         ("EXTRA=", None, "<p><code>EXTRA</code> is appended to every job's overrides.</p>"),
         ("if [ -z \"${NGPU", "Which GPUs",
          """<p>Counts GPUs with <code>nvidia-smi</code> (4 if it cannot), or uses an explicit list in
<code>GPUS</code>. Each job's log goes to <code>output/$RUN/logs/</code>, named with this
invocation's time stamp.</p>"""),
         ("# ---- run the commands", "run_pool(): a tiny job scheduler",
          """<p>Takes the list <code>JOBS</code> ("log name|overrides") and keeps every GPU busy:</p>
<ol>
<li>Make a list of free slots (each GPU × jobs per GPU).</li>
<li>While a slot is free and jobs remain, start the next job in the background on that GPU
(<code>CUDA_VISIBLE_DEVICES</code>), retrying up to RETRIES times and keeping each failed log.</li>
<li>Every 3 seconds, check which jobs finished and free their slots.</li>
</ol>"""),
         ("echo \"======", "Banner", "<p>Prints what is about to run.</p>"),
         ("vname()", None, "<p>Turns <code>true/false</code> into the folder name <code>weighted/unweighted</code>.</p>"),
         ("# ---- phase 1", "Phase 1: train",
          """<p>One job per (process, variant, d, K), smallest cells first so that early results
appear quickly. With more than one job per GPU the per-seed CUDA streams are switched off (see
<a href="core.html#run_optimizers">run_optimizers</a>).</p>"""),
         ("if [ \"$TRAIN_ONLY\"", None, "<p>Optionally stop after training.</p>"),
         ("# ---- phase 2", "Phase 2: evaluate",
          """<p>One job per (process, variant, T_true, d), biggest d first (longest jobs start early),
three per GPU. Each writes only its cell files (<code>eval.part=true</code>).</p>"""),
         ("# ---- phase 3", "Phase 3: merge and draw",
          "<p>One light job per (process, variant, T_true): build results.json and the figures.</p>"),
     ]},
    {"path": "scripts/slurm/submit.sh", "intro_title": "Submitting on a SLURM cluster",
     "intro": """<p>Submits the training array, then the evaluation array with
<code>--dependency=afterok</code> so that it starts only when every training task has succeeded.
Both share one RUN_ID.</p>"""},
    {"path": "scripts/slurm/train.sbatch", "intro_title": "Training as a job array",
     "intro": """<p>48 array tasks (2 processes × 6 dimensions × 4 K), at most 8 at a time. The
<code>#SBATCH</code> lines marked <code>&lt;-- site</code> must be adapted to your cluster.</p>""",
     "notes": [("i=$SLURM_ARRAY_TASK_ID", "Task number → cell",
                """<p>The array index is decoded into (process, d, K), K changing fastest: task 0 is
ddim d=2 K=2, task 1 ddim d=2 K=4, …, task 24 is flow d=2 K=2.</p>""")]},
    {"path": "scripts/slurm/eval.sbatch", "intro_title": "Evaluation as a job array",
     "intro": """<p>6 tasks: one per (process, T_true). Each evaluates the whole grid and draws the
figures (<code>stages=[evaluate,visualize]</code>).</p>"""},
    {"path": "scripts/slurm/README.md", "intro_title": "Cluster instructions",
     "intro": """<div class="callout warn"><p>The "Outputs" paragraph still says models go to
<code>data/&lt;process&gt;/…</code>; the code now uses <code>checkpoints/&lt;process&gt;/…</code>.</p></div>"""},
]

# =============================================================================================
# tests
# =============================================================================================
TEST_FILES = [
    {"path": "code/unit_tests/run.py", "intro_title": "Running the tests",
     "intro": """<div class="formula">python code/unit_tests/run.py</div>
<p>Finds every <code>test_*.py</code> file and runs it with Python's built-in
<code>unittest</code>; no pytest needed. Exits with an error code if anything fails.</p>"""},
    {"path": "code/unit_tests/__init__.py", "intro_title": "Making code/ importable",
     "intro": "<p>Adds <code>code/</code> to the import path so the tests can <code>import train</code> etc.</p>"},
    {"path": "code/unit_tests/_cfg.py", "intro_title": "A tiny configuration",
     "intro": """<p>The real configuration, shrunk so each test finishes in seconds on a CPU: d = 2,
K = 2, 150 training steps, 50 sampler steps, 30 anchors, 300 evaluation seeds per mode. Everything
is written under a temporary folder.</p>"""},
    {"path": "code/unit_tests/test_resume.py",
     "intro_title": "What the tests prove",
     "intro": """<p>All tests are about one promise: <b>however you split up the work, you get the same
files.</b></p>""",
     "notes": [
         ("def _run", "Helpers",
          """<p><code>_run</code> does what main.py does (settings check, train, evaluate).
<code>_results</code> reads results.json without the timing fields; <code>_same_weights</code>
compares two checkpoints tensor by tensor.</p>"""),
         ("def test_pieces_equal_one_run", "Pieces = one run",
          """<p>One invocation with 2 seeds and 2 budgets, against the same run built in three
pieces (seed 0 at one budget, then the other budget, then seed 100). Results and model weights must
be identical, and history.log must have one line per piece.</p>"""),
         ("def test_two_machines_combine_to_one_run", "Two machines = one run",
          """<p>d = 2 on machine A; d = 4 plus another budget of d = 2 on machine B. After combine.py the
results equal a single run that did both.</p>"""),
         ("def test_combine_conflict_keeps_one_model", "A conflict keeps one model",
          """<p>B's model is deliberately altered. combine.py must leave A's model untouched and must not
import B's result rows for that cell.</p>"""),
         ("def test_combine_refuses_other_settings", None, "<p>Different training steps on the two sides → combine refuses.</p>"),
         ("def test_changed_settings_refused", "The settings lock",
          """<p>Choosing other cells, seeds, budgets, variant or T_true is accepted; changing the
training steps or a predictor's degree is refused; <code>strict_run=false</code> allows it on
purpose.</p>"""),
         ("@unittest.skipUnless", "Grouping seeds changes nothing (GPU only)",
          """<p>Two seeds trained together in one CUDA graph must give bit-identical weights to each
seed trained alone: the private random streams of
<a href="core.html#step_generator">step_generator</a> at work.</p>"""),
     ]},
]

# =============================================================================================
PAGES = [
    {"slug": "index", "short": "Start", "title": "The seed-fate atlas, explained",
     "kicker": "iclr_code · guided tour",
     "lede": "Which random starting points make a diffusion model hallucinate, and can we tell in "
             "advance? What the code does, file by file, with every line of it shown.",
     "intro": INDEX_INTRO},
    {"slug": "main", "short": "main & runstate", "title": "Starting a run: main.py and runstate.py",
     "kicker": "entry point", "lede": "How a command turns into stages, and why a run's settings are locked.",
     "files": MAIN_FILES},
    {"slug": "config", "short": "Settings", "title": "Settings and setup: conf/",
     "kicker": "configuration", "lede": "Every knob of the experiment, where it lives and what it does.",
     "intro": CONFIG_INTRO, "files": CONFIG_FILES},
    {"slug": "core", "short": "core.py", "title": "The maths: core.py",
     "kicker": "building block", "lede": "The mixture, the fate rule, the network, the training recipe, "
     "the exact sampler and the anchors.", "files": [CORE_FILE]},
    {"slug": "processes", "short": "processes/", "title": "DDIM and flow matching: processes/",
     "kicker": "building block", "lede": "Two ways of turning noise into data, behind one interface.",
     "files": PROC_FILES},
    {"slug": "train", "short": "train.py", "title": "Stage 1: training the learned samplers",
     "kicker": "pipeline · stage 1", "lede": "One network per cell, retrained if it hallucinates too "
     "often, saved with everything needed to evaluate it.", "files": [TRAIN_FILE]},
    {"slug": "evaluate", "short": "evaluate.py", "title": "Stage 2: building and scoring the atlas",
     "kicker": "pipeline · stage 2", "lede": "Anchors, backtracking, predictors and scores, stored per "
     "cell so that nothing is ever computed twice.", "files": [EVAL_FILE]},
    {"slug": "fate", "short": "fate.py", "title": "Predicting a seed's fate: fate.py",
     "kicker": "building block", "lede": "Four predictors and the scores they are judged by.",
     "files": [FATE_FILE]},
    {"slug": "visualize", "short": "visualize.py", "title": "Stage 3: figures and tables",
     "kicker": "pipeline · stage 3", "lede": "Heatmaps over the (d, K) grid and a table for the paper, "
     "from results.json alone.", "files": [VIZ_FILE]},
    {"slug": "combine", "short": "combine.py", "title": "Merging runs from several machines",
     "kicker": "tool", "lede": "Bring a run computed elsewhere into this one without ever mixing two "
     "models' results.", "files": [COMBINE_FILE]},
    {"slug": "rank_probe", "short": "rank_probe.py", "title": "Side experiment: are hallucinations low-rank?",
     "kicker": "tool", "lede": "A stand-alone script that measures how many dimensions the "
     "hallucinating seeds really occupy.", "files": [RANK_FILE]},
    {"slug": "scripts", "short": "scripts/", "title": "Running the full sweep: scripts/",
     "kicker": "operations", "lede": "Spreading thousands of jobs over GPUs, or over a SLURM cluster.",
     "files": SCRIPT_FILES},
    {"slug": "tests", "short": "tests", "title": "The tests: unit_tests/",
     "kicker": "quality", "lede": "Proof that resuming, splitting and merging a run never changes its results.",
     "files": TEST_FILES},
]
NAV = [["index"], ["config", "main"], ["core", "processes"], ["train", "evaluate", "fate", "visualize"],
       ["combine", "rank_probe"], ["scripts", "tests"]]
BY_SLUG = {p["slug"]: p for p in PAGES}
assert sorted(s for g in NAV for s in g) == sorted(BY_SLUG), "NAV and PAGES disagree"


def prepare():
    """Compute the figures and fill the placeholders (called by build.py)."""
    svg, f = diagrams.fate_map(T=500)
    cap = (f"<b>Computed, not drawn.</b> d = 2, K = 3, σ = 0.1, the mode centres from "
           f"<code>core.sample_modes(seed=0)</code>, the exact DDIM sampler with T = {f['T']} steps "
           f"(<code>core.forward_true</code>), fates from <code>core.label_fate</code>. "
           f"<b>Left:</b> seed space is carved into one region per mode. Seeds far from the origin "
           f"(outside the dotted circles, rarely drawn from N(0, I)) end up in no ball. "
           f"<b>Right:</b> {f['n_seeds']} seeds drawn from N(0, I); {f['n_hall_shown']} of them (✕) "
           f"land outside every R99 ball. Over 200 000 seeds the hallucination rate here is "
           f"{f['hall_rate'] * 100:.1f}%. The atlas tries to learn the left picture in up to 512 dimensions, "
           f"where it cannot be drawn.")
    cards = "".join(f'<a class="card" href="{slug}.html"><b>{BY_SLUG[slug]["short"]}</b>'
                    f'<code>{files}</code><br><span>{what}</span></a>' for slug, files, what in CARDS)
    BY_SLUG["index"]["intro"] = (INDEX_INTRO.replace("{{FATE_MAP}}", svg)
                                 .replace("{{FATE_CAPTION}}", cap).replace("{{CARDS}}", cards))
    for i, (a, t, body) in enumerate(CORE_FILE["notes"]):
        if "{{ANCHORS}}" in body:
            CORE_FILE["notes"][i] = (a, t, body.replace("{{ANCHORS}}", diagrams.anchors()))
