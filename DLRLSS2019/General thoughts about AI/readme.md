#  ML for AI: What's next? by Yoshua Bengio
### 1.1 AI current: from from human-level AI
* sample complexity
* shallow and low-level 'understanding'
### 1.2 Unpromising researches towards human-level AI
* NLP based purely on text
* Generative models purely on sensary data (pixel level)
* Prior knowledge
* Unscaleble algorithms
* Theories not compatible with animal/human situation, real environment
### 1.3 Promising researches
* Example: High-level factors is much easier to generalize --> Grounded language learning, jointly learning Natural Language and a World Model --> BabyAI platform (ICLR 2019)
* Integrating system 1 and system 2, like multimodal 
<p align="center">
	<img src="http://upfrontanalytics.com/SITE/wp-content/uploads/2015/04/System-1-vs-System-2.jpg" alt="Sample"  width="250" height="200">
	<p align="center">
		<em> system 1 and system 2 </em>
	</p>
</p>

* Generative model in Latent space (high-level abstract space) --> unsupervised learning + self-supervised learning (denoising auto-encoders, BERT, Word2Vec, XLNet) --> Spatio-Temporal Deep Infomax(STDIM, ICLR2019 paper) --> generate on consciousness prior (2 levels of representation)
* Attention mechanism! attention to learn waht to memorize from past, what to predict abouth future --> predict some given very a few (soarse factor graph)
* [Represntation learning](http://www.iro.umontreal.ca/~lisa/pointeurs/TPAMISI-2012-04-0260-1.pdf)
* Unsupervised agnecy with intrisic rewards (curiosity, controllability)
* Transfer learning / Meta-learning, reuse, fast adaption to changes in distribution
* Breaking knowledge into recomposable pieces:reusable pirces
* Causality, correct causal structure leads to faster adaption --> turn nonstationarities in distribution into factorize knowledge to maximize fast transfer [paper](https://kopernio.com/viewer?doi=arXiv:1901.10912&route=6)
### 1.4 AI and society
...

# AI thoughts and RL frontiers by Richard Sutton
### 2.1 About research
* Some notes:<br>
 Don't be impressed by what you don't understand<br>
 Don't try to impress others by what they don't understand<br>
 You should be brave and ambitious but also humble and transparent<br>
 Humble before the great task (e.g., understanding the mind。<br>
* **How to train yourself to think carefully and productively?**  to write is to think (discuss with yourself)
* When get stuck, persist and try:
 Define your terms <br>
 Go multiple (i.e., what are some of the conceivable answers?) <br>
 Go meta (i.e., that would an answer look like? what properties would it have?) <br>
 Retreat (retreat to a clearer question that you can make progress on, then come back)<br>
* The most important insight you will ever contribute is 1. propably something that you already know; 2. probably something that is obvious to you. <br> Some examples: <br>
 No animal does supervised learning<br>
 No mind generates images or videos<br>
 Neural networks are not in any meaningful sense "neural"<br>
 People are machines<br>
 The purpose of life is pleasure<br>
 The world is much more complex than any mind that tries to understand it (therefore, a prior distribution on the world counld never be reasonable)<br>
 Mind is computational, and computation is increasing exponentially<br>
 Human input doesn't scale, the only scable methods are search and learning<br>
* Some others:<br>
 Experience is the data of AI<br>
 Approximate the solution, not the problem<br>
 Take the agent's point of view<br>
 Set measurable goals for the subparts of an agent<br>
 Work by orthogonal dimensions, work issue by issue<br>
 Work on ideas, not software<br>
* Exercice 1. try to define **What is Intelligence**.
* Exercise 2. try to figure out the exceptions of the **Predictive knowledge hypothesis**.

### 2.2 Tricks for doing RL research
Know where the frontier is
![Frontiers](https://github.com/HIT-SMC/Public-Presentations/blob/master/DLRLSS2019/General%20thoughts%20about%20AI/trick-1.jpg)
Extend the frontier by several dimensions and comple the square.
![Extend the frontiers](https://github.com/HIT-SMC/Public-Presentations/blob/master/DLRLSS2019/General%20thoughts%20about%20AI/trick-2.jpg)
### 2.3 RL researches that may be interesting
* A possible new view of the ML landscape: prediction learning, control learning, representation learning, and integrated agent architectures
* Core RL research learns value function and policies (value-based/ policy-based RL), next we need to learn (State features, skills, subproblems, and model of the world in Part II of [RL: and introduction](http://incompleteideas.net/book/RLbook2018trimmed.pdf))
* Animals and babies play to pursue subproblems <br> Three open questions/
 What should the subproblems be?<br>
 Where do the subproblems come from?<br>
 How do the subproblems help on the main problem?<br>
* Permanent and transient memories
* Big world vs small mind, apparent non-stationary --> changing approximate value function
### 2.4 AI & Society
AI will inevitably lead to new beings and new ways of beings that are much more powerful than our current selves.
