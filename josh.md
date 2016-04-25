Recent efforts to creating dispatching rules have focused on direct
search methods and learning from scheduling data. This paper will
examine the latter approach and present a systematic approach for doing
so effectively. The key to learning an effective dispatching rule is
through the careful construction of the training data,
$\{\vec{x}_i(k),y_i(k)\}_{k=1}^K\in\mathcal{D}$, where

features of partially constructed schedules $\vec{x}_i$ should
necessarily reflect the induced data distribution $\mathcal{D}$ for when
the rule is applied. This is achieved by updating the learned model in
an active imitation learning fashion

$y_i$ is labelled optimally using a MIP solver

data needs to be balanced, as the set is unbalanced w.r.t. dispatching
step $k$

When querying an optimal policy, there is an abundance of valuable
information that can be utilised for learning new dispatching rules. For
instance, it’s possible to seek out when the scheduling process is most
susceptible to failure. Generally stepwise optimality (or training
accuracy) will imply good end performance, here minimising the makespan.
However, as the impact of suboptimal moves is not fully understood, the
labelling must be adjusted for its intended trajectory.

Using the guidelines set by the framework the design of custom
dispatching rules, for one’s particular scheduling application, will be
more effective. In the study presented three different distributions of
the job-shop will be considered. The machine learning approach
considered is based on preference learning, which learns what post
decision state is preferable to another. However, alternative learning
methods may be applied to the training data generated.

Introduction {#sec:introduction}
============

Hand crafting heuristics for scheduling is an ad-hoc approach to finding
approximate solutions to problems. The practice is time-consuming and
its performance can even vary dramatically between different problem
instances. The aim of this work is to increase our understanding of this
process. In particular the learning of new problem specific priority
dispatching rules (DR) will be addressed for a subclass of scheduling
problems known as the job-shop scheduling problem (JSP).

A recent editorial of the state-of-the-art approaches @Chen13 in
advanced dispatching rules for large-scale manufacturing systems reminds
us that: ... most traditional dispatching rules are based on historical
data. With the emergence of data mining and on-line analytic processing,
dispatching rules can now take predictive information into account. The
importance of automated discovery of dispatching rules was also
emphasised by @Monch13. Data for learning can also be generated using a
known heuristic on a set of problem instances. Such an approach is taken
in @Siggi05 for single-machine where a decision tree is learned from the
data to have similar logic to the guiding dispatching rule. However, the
learned method cannot outperform the original dispatching rule used for
the data generation. This drawback is confronted in @Malik08
[@Russell09; @Siggi10] by using an optimal scheduler or policy, computed
off-line. The resulting dispatching rules, as decision trees, gave
significantly better schedules than using popular heuristics in that
field, and a lower worst-case factor from optimality. Although, using
optimal policies for creating training data gives vital information on
how to learn good scheduling rules an experimental study will show that
this is not sufficient. Once these rules make a suboptimal dispatch then
they are in uncharted territory and its effects are relatively unknown.
This work will illustrate the sensitivity of learned dispatching rule’s
performance on the way the training data is sampled. For this purpose,
JSP is used as a case study to illustrate a methodology for generating
meaningful training data, which can be successfully learned using
preference-based imitation learning (IL).

The competing alternative to learning dispatching rules from data would
be to search the dispatching rule space directly. The prevalent approach
in this case would be using an evolutionary algorithm, such as genetic
programming (GP). The predominant approach in hyper-heuristics is a
framework of creating new heuristics from a set of predefined heuristics
via genetic algorithm optimisation @Burke10. Adopting a two-stage
hyper-heuristic approach to generate a set of machine-specific DRs for
dynamic job-shop, @Pickardt2013 used genetic programming (GP) to evolve
CDRs from basic features, along with evolutionary algorithm to assign a
CDR to a specific machine. The problem space consists of job-shops in
semiconductor manufacturing, with additional shop constraints, as
machines are grouped to similar work centres, which can have different
set-up time, workload, etc. In fact, the GP emphasised on efficiently
dispatching on the work centres with set-up requirements and batching
capabilities, which are rules that are non-trivial to determine
manually.

With meta heuristics one can use existing DRs, and use for example
portfolio-based algorithm selection @Rice76 [@Gomes01; @Xu07], either
based on a single instance or class of instances to determine which DR
to choose from. Implementing ant colony optimisation to select the best
DR from a selection of nine DRs for JSP, experiments from @Korytkowski13
showed that the choice of DR do affect the results and that for all
performance measures considered. They showed that it was better to have
a all the DRs to choose from rather than just a single DR at a time. A
simpler and more straightforward way to automate selection of composite
priority dispatching rules (CDR), @InRu14, translated dispatching rules
into measurable features which describe the partial schedule and
optimise directly what their contribution should be via evolutionary
search.

Using case based reasoning for timetable scheduling, training data in
@Burke06 is guided by the two best heuristics in the literature. They
point out that in order for their framework to be successful, problem
features need to be sufficiently explanatory and training data needs to
be selected carefully so they can suggest the appropriate solution for a
specific range of new cases. When learning new dispatching rules there
are several important factors to consider. First and foremost the
context in which the training data is constructed will influence the
quality of the learned dispatching rule @Burke06. Since the training
data consists of collection of features, the quality of training data is
interchangeable to the predictability of features. The training data is
necessarily also problem instance specific. In addition to addressing
these aspects, the paper will show that during the scheduling process,
it will vary when it is most critical to make the ‘right’ dispatch.
Furthermore, depending on the distribution of problem instances these
critical moments can vary greatly. Moreover, a supervised learning
algorithm will optimize classification accuracy, while it is the actual
end-performance of the dispatching rule learned that will determine the
success of the learning method.

The outline of the paper is the following, gives the mathematical
formalities of the scheduling problem, and describes the main featues
for job-shop, and illustrates how schedules are created with dispatching
rules. sets up the framework for learning from optimal schedules. In
particular, the probability of choosing optimal decisions and the
effects of making a suboptimal decision. Furthermore, the optimality of
common single priority dispatching rules is investigated. With these
guidelines, goes into detail on how to create meaningful composite
priority dispatching rules using preference learning, focusing on how to
compare operations and collect training data with the importance of the
sampling strategy applied. explain the trajectories for sampling
meaningful schedules used in preference learning, either using passive
or active imitation learning. Experimental results are jointly presented
in with comparison for a randomly generated problem space. Furthermore,
some general adjustments for performance boost is also considered. The
paper finally concludes in with discussion and conclusions.

Job-shop Scheduling {#sec:problemdef}
===================

JSP involves the scheduling of jobs on a set of machines. Each job
consists of a number of operations which are then processed on the
machines in a predetermined order. An optimal solution to the problem
will depend on the specific objective.

This study will consider the $n\times m$ JSP, where $n$ jobs,
$\mathcal{J}=\{J_j\}_{j=1}^n$, are scheduled on a finite set,
$\mathcal{M}=\{M_a\}_{a=1}^m$, of $m$ machines. The index $j$ refers to
a job $J_j\in\mathcal{J}$ while the index $a$ refers to a machine
$M_a\in\mathcal{M}$. Each job requires a number of processing steps or
operations, the pair $(j,a)$ refers to the operation, i.e., processing
the task of job $J_j$ on machine $M_a$.

Each job $J_j$ has an indivisible operation time (or cost) on machine
$M_a$, $p_{ja}$, which is assumed to be integral and finite. Starting
time of job $J_j$ on machine $M_a$ is denoted $x_s(j,a)$ and its end
time is denoted $x_e(j,a)$ where:

$$\quad x_e(j,a):=x_s(j,a)+p_{ja}$$

Each job $J_j$ has a specified processing order through the machines. It
is a permutation vector, $\bm \sigma_j$, of $\{1,\ldots,m\}$,
representing a job $J_j$ can be processed on $M_{\bm \sigma_j(a)}$ only
after it has been completely processed on $M_{\bm \sigma_j(a-1)}$,
namely:

$$\quad\label{eq:permutation}
x_s(j,\bm \sigma_j(a)) \geq x_e(j,\bm \sigma_j(a-1))$$

for all $J_j\in\mathcal{J}$ and $a\in\{2,..,m\}$. Note, that each job
can have its own distinctive flow pattern through the machines, which is
independent of the other jobs. However, in the case that all jobs share
the same fixed permutation route, it is referred to as flow-shop (FSP).
A commonly used subclass of FSP in the literature is permutation
flow-shop, which has the added constraint that the processing order of
the jobs on the machines must be identical as well, i.e., no passing of
jobs allowed @Stafford88.

The disjunctive condition that each machine can handle at most one job
at a time is the following:

$$\quad\label{eq:oneJobPerMac}
x_s(j,a) \geq x_e(j',a) \quad\textrm{or}\quad x_s(j',a) \geq x_e(j,a)$$

for all $J_j,J_{j'}\in\mathcal{J},\; J_j\neq J_{j'}$ and
$M_a\in\mathcal{M}$.

The objective function is to minimise the schedule’s maximum completion
times for all tasks, commonly referred to as the makespan, $C_{\max}$,
which is defined as follows:

$$\quad
C_{\max} := 
\max\left\{x_e(j,\bm \sigma_j(m))\;:\;J_j\in\mathcal{J}\right\}.\label{eq:makespan}$$

This family of scheduling problems is denoted by $J||C_{\max}$
@Pinedo08. Additional constraints commonly considered are job
release-dates and due-dates or sequence dependent set-up times, however,
these will not be considered here.

In order to find an optimal (or near optimal) solution for scheduling
problems one could either use exact methods or heuristics methods. Exact
methods guarantee an optimal solution. However, job-shop scheduling is
strongly NP-hard @Garey76:NPhard. Any exact algorithm generally suffers
from the curse of dimensionality, which impedes the application in
finding the global optimum in a reasonable amount of time. Using
state-of-the-art software for solving scheduling problems, such as LiSA
(A Library of Scheduling Algorithms) @LiSA, which includes a specialised
version of branch and bound that manages to find optimums for
job-shop problems of up to $14\times14$ @Ru12. However, problems that
are of greater size, become intractable. Heuristics are generally more
time efficient but do not necessarily attain the global optimum.
Therefore, job-shop has the reputation of being notoriously difficult to
solve. As a result, it’s been widely studied in deterministic scheduling
theory and its class of problems has been tested on a plethora of
different solution methodologies from various research fields @Meeran12,
all from simple and straight forward dispatching rules to highly
sophisticated frameworks.

Priority Dispatching Rules {#sec:DR}
==========================

Priority dispatching rules determine, from a list of incomplete jobs,
$\mathcal{L}$, which job should be dispatched next. This process, where
an example of a temporal partial schedule of six-jobs scheduled on
five-machines, is illustrated in . The numbers in the boxes represent
the job identification $j$. The width of the box illustrates the
processing times for a given job for a particular machine $M_a$ (on the
vertical axis). The dashed boxes represent the resulting partial
schedule for when a particular job is scheduled next. Moreover, the
current $C_{\max}$ is denoted by a dotted vertical line. The object is
to keep this value as small as possible once all operations are
complete. As shown in the example there are $15$ operations already
scheduled. The *sequence* of dispatches used to create this partial
schedule is:

$$\quad
\bm \chi=\left(J_3,J_3,J_3,J_3,J_4,J_4,J_5,J_1,J_1,J_2,J_4,J_6,J_4,J_5,J_3\right)$$

This refers to the sequential ordering of job dispatches to machines,
i.e., $(j,a)$; the collective set of allocated jobs to machines is
interpreted by its sequence which is referred to as a schedule. A
scheduling policy will pertain to the manner in which the sequence is
determined from the available jobs to be scheduled. In our example, the
available jobs are given by the job-list
$\mathcal{L}^{(k)}=\{J_1,J_2,J_4,J_5,J_6\}$ with the five potential jobs
to be dispatched at step $k=16$ (note that $J_3$ is completed).

However, deciding which job to dispatch is not sufficient as one must
also know where to place it. In order to build tight schedules it is
sensible to place a job as soon as it becomes available and such that
the machine idle time is minimal, i.e., schedules are non-delay. There
may also be a number of different options for such a placement. In one
observes that $J_2$, to be scheduled on $M_3$, could be placed
immediately in a slot between $J_3$ and $J_4$, or after $J_4$ on this
machine. If $J_6$ had been placed earlier, a slot would have been
created between it and $J_4$, thus creating a third alternative, namely
scheduling $J_2$ after $J_6$. The time in which machine $M_a$ is idle
between consecutive jobs $J_j$ and $J_{j'}$ is called idle time or
slack:

$$\quad 
s(a,j):=x_s(j,a)-x_e(j',a) \label{eq:slack}$$

where $J_j$ is the immediate successor of $J_{j'}$ on $M_a$.

Construction heuristics are designed in such a way that it limits the
search space in a logical manner respecting not to exclude the optimum.
Here, the construction heuristic, $\Upsilon$, is to schedule the
dispatches as closely together as possible, i.e., minimise the
schedule’s idle time. More specifically, once an operation $(j,a)$ has
been chosen from the job-list $\mathcal{L}$ by some dispatching rule, it
can then be placed immediately after (but not prior) to
$x_e(j,\bm \sigma_j(a-1))$ on machine $M_a$ due to constraint . However,
to guarantee that constraint is not violated, idle times $M_a$ are
inspected as they create a slot which $J_j$ can occupy. Bearing in mind
that $J_j$ release time is $x_e(j,\bm \sigma_j(a-1))$ one cannot
implement directly, instead it has to be updated as follows:

$$\quad
\tilde{s}(a,j'):= x_s(j'',a)-\max\{x_e(j',a),x_e(j,\bm \sigma_j(a-1))\}$$

for all already dispatched jobs, $J_{j'},J_{j''}\in \mathcal{J}_a$ where
$J_{j''}$ is $J_{j'}$ successor on $M_a$. Since preemption is not
allowed, the only applicable slots are whose idle time can process the
entire operation, namely:

$$\quad
\tilde{S}_{ja} := \left\{J_{j'}\in \mathcal{J}_a \;:\; \tilde{s}(a,j')\geq 
p_{ja} 
\right\}\label{eq:slots}.$$

The placement rule applied will decide where to place the job and is
intrinsic to the construction heuristic, which is chosen independently
of the priority dispatching rule that is applied. Different placement
rules could be considered for selecting a slot from , e.g., if the main
concern were to utilise the slot space, then choosing the slot with the
smallest idle time would yield a closer-fitted schedule and leave
greater idle times undiminished for subsequent dispatches on $M_a$. In
our experiments, cases were discovered where such a placement could rule
out the possibility of constructing optimal solutions. However, this
problem did not occur when jobs are simply placed as early as possible,
which is beneficial for subsequent dispatches for $J_j$. For this
reason, it will be the placement rule applied here.

![image](figures/jssp_example) [fig:jssp:example]

Priority dispatching rules will use features of operations, such as
processing time, in order to determine the job with the highest
priority. Consider again , if the job with the shortest processing time
(SPT) were to be scheduled next, then $J_2$ would be dispatched.
Similarly, for the longest processing time (LPT) heuristic, $J_5$ would
have the highest priority. Dispatching can also be based on features
related to the partial schedule. Examples of these are dispatching the
job with the most work remaining (MWR) or alternatively the least work
remaining (LWR). A survey of more than $100$ of such rules are presented
in @Panwalkar77. However, the reader is referred to an in-depth survey
for simple or *single priority dispatching rule* (SDR) by @Haupt89. The
SDRs assign an index to each job in the job-list and is generally only
based on a few features and simple mathematical operations.

[t!] [tbl:jssp:feat]

<span>cll</span> $\bm{\phi}$ & Feature description & Mathematical
formulation\
\
$\phi_1$& job processing time & $p_{ja}$\
$\phi_2$& job start-time & $x_s(j,a)$\
$\phi_3$& job end-time & $x_e(j,a)$\
$\phi_4$& job arrival time & $x_e(j,a-1)$\
$\phi_5$& time job had to wait & $x_s(j,a)-x_e(j,a-1)$\
$\phi_6$& total processing time for job &
$\sum_{a\in \mathcal{M}}p_{ja}$\
$\phi_7$& total work remaining for job &
$\sum_{a'\in\mathcal{M}\setminus \mathcal{M}_{j}}p_{ja'}$\
$\phi_8$& number of assigned operations for job & $|\mathcal{M}_j|$\
\
$\phi_{9}$& when machine is next free &
$\max_{j'\in \mathcal{J}_a} \{x_e(j',a)\}$\
$\phi_{10}$& total processing time for machine &
$\sum_{j\in \mathcal{J}}p_{ja}$\
$\phi_{11}$& total work remaining for machine &
$\sum_{j'\in\mathcal{J}\setminus \mathcal{J}_{a}}p_{j'a}$\
$\phi_{12}$& number of assigned operations for machine &
$|\mathcal{J}_a|$\
$\phi_{13}$& change in idle time by assignment & $\Delta s(a,j)$\
$\phi_{14}$& total idle time for machine &
$\sum_{j'\in \mathcal{J}_a}s(a,j')$\
$\phi_{15}$& total idle time for all machines & $\sum_{a'\in 
	\mathcal{M}}\sum_{j'\in \mathcal{J}_{a'}}s(a',j')$\
$\phi_{16}$& current makespan &
$\max_{(j',a')\in \mathcal{J} \times \mathcal{M}_{j'}}\{x_f(j',a')\}$\

Designing priority dispatching rules requires recognising the important
features of the partial schedules needed to create a reasonable
scheduling rule. These features attempt to grasp key attributes of the
schedule being constructed. Which features are most important will
necessarily depend on the objectives of the scheduling problem. Features
used in this study applied for each possible operation encountered are
given in , where the set of machines already dispatched for $J_j$ is
$\mathcal{M}_j\subset\mathcal{M}$, and similarly, $M_a$ has already had
the jobs $\mathcal{J}_a\subset\mathcal{J}$ previously dispatched. The
features of particular interest were obtained by inspecting the
aforementioned SDRs. Features $\phi_1$-$\phi_8$ and
$\phi_{9}$-$\phi_{16}$ are job-related and machine-related,
respectively. In fact, @Pickardt2013 note that in the current
literature, there is a lack of global perspective in the feature space,
as omitting them won’t address the possible negative impact an operation
$(j,a)$ might have on other machines at a later time, it is for that
reason features such as $\phi_{13}$-$\phi_{15}$ are considered, since
they are slack related and are a means of indicating the current quality
of the schedule. All of the features, $\bm{\phi}$, vary throughout the
scheduling process, w.r.t. operation belonging to the same time step
$k$, with the exception of $\phi_6$ and $\phi_{10}$ which are static for
a given problem instance but varying for each $J_j$ and $M_a$,
respectively.

Priority dispatching rules are attractive since they are relatively easy
to implement, perform fast, and find reasonable schedules. In addition,
they are relatively easy to interpret, which makes them desirable for
the end-user. However, they can also fail unpredictably. A careful
combination of dispatching rules has been shown to perform significantly
better @Jayamohan04. These are referred to as *composite priority
dispatching rules* (CDR), where the priority ranking is an expression of
several dispatching rules. CDRs deal with a greater number of more
complicated functions and are constructed from the schedules features.
In short, a CDR is a combination of several DRs. For instance let $\pi$
be a CDR comprised of $d$ DRs, then the index $I$ for
$J_j\in\mathcal{L}^{(k)}$ using $\pi$ is:

$$\quad I_j^{\pi} = \sum_{i=1}^d w_i \pi_i(\bm \chi^j) 
\label{eq:CDR}$$

where $w_i>0$ and $\sum_{i=0}^d w_i = 1$ with $w_i$ giving the weight of
the influence of $\pi_i$ (which could be a SDR or another CDR) to $\pi$.
Note: each $\pi_i$ is a function of $J_j$’s features from the current
sequence $\bm \chi$, where $\bm \chi^j$ implies that $J_j$ was the
latest dispatch, i.e., the partial schedule given $\chi_k=J_j$.

At each step $k$, an operation is dispatched which has the highest
priority. If there is a tie, some other priority measure is used.
Generally the dispatching rules are static during the entire scheduling
process. However, ties could also be broken randomly (RND).

While investigating 11 SDRs for JSP, @Lu13 a pool of 33 CDRs was
created. This pool strongly outperformed the original CDRs by using
multi-contextual functions based on either job waiting time or machine
idle time (similar to $\phi_5$ and $\phi_{14}$ in ), i.e., the CDRs are
a combination of either one or both of these key features and then the
SDRs. However, there are no combinations of the basic SDRs explored,
only those two features. Similarly, using priority rules to combine 12
existing DRs from the literature, @Yu13 had 48 CDR combinations which
yielded 48 different models to implement and test. It is intuitive to
get a boost in performance by introducing new CDRs, since where one DR
might be failing, another could be excelling, so combining them together
should yield a better CDR. However, these approaches introduce fairly
ad-hoc solutions and there is no guarantee the optimal combination of
dispatching rules are found.

The composite priority dispatching rule presented in can be considered
as a special case of a the following general linear value function:

$$\quad\label{eq:jssp:linweights}
\pi(\bm \chi^j)=\sum_{i=1}^d w_i \phi_i(\bm \chi^j).$$

when $\pi_i(\cdot)=\phi_i(\cdot)$, i.e., a composite function of the
features from . Finally, the job to be dispatched, $J_{j^*}$,
corresponds to the one with the highest value, namely:

$$\quad\label{eq:jstar}
J_{j^*}=\mathop{\rm argmax}_{J_j\in \mathcal{L}}\; \pi(\bm \chi^j)$$

Similarly, single priority dispatching rules may be described by this
linear model. For instance, let all $w_i=0$, but with following
exceptions: $w_1=-1$ for SPT, $w_1=+1$ for LPT, $w_7=-1$ for LWR and
$w_7=+1$ for MWR. Generally, the weights $\vec{w}$ are chosen by the
designer or the rule apriori. A more attractive approach would be to
learn these weights from problem examples directly. The following will
investigate how this may be accomplished.

Performance Analysis of Priority Dispatching Rules {#sec:learnOPT}
==================================================

In order to create successful dispatching rules, a good starting point
is to investigate the properties of optimal solutions and hopefully be
able to learn how to mimic the construction of such solutions. For this,
optimal solutions (obtained by using a commercial software package
@gurobi) are followed and the probability of SDRs being optimal is
inspected. This serves as an indicator of how hard it is to put our
objective up as a machine learning problem. However, the end-goal, which
is minimising deviation from optimality, $\rho$, must also take into
consideration because of its relationship to stepwise optimality is not
fully understood.

In this the concerns of learning new priority dispatching rules will be
addressed. At the same time experimental set-up used in the study are
described.

Problem Instances {#sec:data:sim}
-----------------

The class of problem instances used in our studies is the
job-shop scheduling problem described in . Each instance will have
different processing times and machine ordering. Each instance will
therefore create different challenges for a priority dispatching rule.
Dispatching rules learned will be customised for the problems used for
their training. For real world application using historical data would
be most appropriate. The aim would be to learn a dispatching rule that
works well on average for a given distribution of problem instances. To
illustrate the performance difference of priority dispatching rules on
different problem distributions within the same class of problems,
consider the following three cases. Problem instances for JSP are
generated stochastically by fixing the number of jobs and machines to
ten. A discrete processing time is sampled independently from a discrete
uniform distribution from the interval $I=[u_1,u_2]$, i.e.,
$\vec{p}\sim \mathcal{U}(u_1,u_2)$. The machine order is a random
permutation of all of the machines in the job-shop. Two different
processing times distributions were explored, namely
$\mathcal{P}_{j.rnd}^{n \times m}$ where $I=[1,99]$ and
$\mathcal{P}_{j.rndn}^{n \times m}$ where $I=[45,55]$. These instances
are referred to as random and random-narrow, respectively. In addition,
the case where the machine order is fixed and the same for all jobs,
i.e. $\sigma_j(a)=a$ for all $J_j\in\mathcal{J}$ and where
$\vec{p}\sim\mathcal{U}(1,99)$, is also considered. These jobs are
denoted by $\mathcal{P}_{f.rnd}^{n \times m}$ and are analogous to
$\mathcal{P}_{j.rnd}^{n \times m}$. The problem spaces are summarised in
.

The goal is to minimise the makespan, $C_{\max}$. The optimum makespan
is denoted $C_{\max}^{\pi_\star}$ (using the expert policy $\pi_\star$),
and the makespan obtained from the scheduling policy $\pi$ under
inspection by $C_{\max}^{\pi}$. Since the optimal makespan varies
between problem instances the performance measure is the following:

$$\quad\label{eq:rho}
\rho=\frac{C_{\max}^{\pi}-C_{\max}^{\pi_\star}}{C_{\max}^{\pi_\star}}\cdot
100\%$$

which indicates the percentage relative deviation from optimality. Note:
measures the discrepancy between predicted value and true outcome, and
is commonly referred to as a loss function, which should be minimised
for policy $\pi$.

depicts the box-plot for when using the SDRs from for all of the problem
spaces from . These box-plots show the difference in performance of the
various SDRs. The rule MWR performs on average the best on the
$\mathcal{P}_{j.rnd}^{n \times m}$ and
$\mathcal{P}_{j.rndn}^{n \times m}$ problems instances, whereas for
$\mathcal{P}_{f.rnd}^{n \times m}$ it is LWR that performs best. It is
also interesting to observe that all but MWR perform statistically worse
than a random job dispatching on the $\mathcal{P}_{j.rnd}^{n \times m}$
and $\mathcal{P}_{j.rndn}^{n \times m}$ problems instances.

[tbl:data:sim]

<span>lcccl</span>name & size ($n\times m$) & $N_{\text{train}}$ &
$N_{\text{test}}$ & note\
$\mathcal{P}_{j.rnd}^{10 \times 10}$ & $10\times10$ & 300 & 200 &
random\
$\mathcal{P}_{j.rndn}^{10 \times 10}$ & $10\times10$ & 300 & 200 &
random-narrow\
$\mathcal{P}_{f.rnd}^{10 \times 10}$ & $10\times10$ & 300 & 200 &
random\

![Box-plot for deviation from optimality, $\rho$, (%) for
SDRs](figures/{boxplotRho_SDR_10x10}.pdf "fig:") [fig:boxplot:SDR]

Reconstructing optimal solutions {#sec:opt:sdr}
--------------------------------

When building a complete schedule, $K=n\cdot m$ dispatches must be made
sequentially. A job is placed at the earliest available time slot for
its next machine, whilst still fulfilling that each machine can handle
at most one job at each time, and jobs need to have finished their
previous machines according to their machine order. Unfinished jobs from
the job-list $\mathcal{L}$ are dispatched one at a time according to a
deterministic scheduling policy (or heuristic). This process is given as
a pseudo-code is given in . After each dispatch[^1] the schedule’s
current features are updated based on the half-finished schedule,
$\bm \chi$. For each possible post-decision state the temporal features
are collected (cf. ) forming the feature set, $\Phi$, based on all
$N_{\text{train}}$ problem instances available, namely:

$$\quad \label{eq:Phi}
\Phi := \bigcup_{\{\vec{x}_i\}_{i=1}^{N_{\text{train}}}} 
\left\{\bm{\phi}^j \;:\; J_j\in\mathcal{L}^{(k)}\right\}_{k=1}^K
\subset\mathcal{F}$$

where the feature space $\mathcal{F}$ is described in , and are based on
job- and machine-features which are widespread in practice.

[t] [pseudo:constructJSP]

[1] $\bm \chi\gets \emptyset$
$\bm{\phi}^j \gets \bm{\phi}\circ\Upsilon\left(\bm \chi^j\right)$
[pseudo:constructJSP:phi] $I_j^{\pi} \gets \pi\left(\bm{\phi}^j\right)$
$j^* \gets \mathop{\rm argmax}_{j\in \mathcal{L}^{(k)}}\{I_j^{\pi}\}$
$\chi_k \gets J_{j^*}$ $C_{\max}^{\pi} \gets \Upsilon(\bm \chi)$

It is easy to see that the sequence of task assignments is by no means
unique. Inspecting a partial schedule further along in the dispatching
process such as in , then let’s say $J_1$ would be dispatched next, and
in the next iteration $J_2$. Now this sequence would yield the same
schedule as if $J_2$ would have been dispatched first and then $J_1$ in
the next iteration, i.e., these are jobs with non-conflicting machines.
In this particular scenario, one cannot infer that choosing $J_1$ is
better and $J_2$ is worse (or vice versa) since they can both yield the
same solution. Furthermore, there may be multiple optimal solutions to
the same problem instance. Hence not only is the sequence representation
‘flawed’ in the sense that slight permutations on the sequence are in
fact equivalent w.r.t. the end-result, but very varying permutations on
the dispatching sequence (although given the same partial initial
sequence) can result in very different complete schedules yet can still
achieve the same makespan.

The redundancy in building optimal solutions using dispatching rules
means that many different dispatches may yield an optimal solution to
the problem instance. Let’s formalise the probability of optimality (or
stepwise classification accuracy) for a given policy $\pi$, as:

$$\quad \label{eq:tracc:opt}
\xi^\star_{\pi} := \mathbb{E}_{\pi_\star}\left\{\pi_{\star} = \pi \right\}$$

that is to say the mean likelihood of our policy $\pi$ being equivalent
to the expert policy $\pi_\star$. The probability that a job chosen by a
SDR yields an optimal makespan on a step-by-step basis, i.e.,
$\xi^\star_{\langle \text{SDR} \rangle}$, is depicted in . These
probabilities vary quite a bit between the different problem instances
distributions studied. From it is observed that $\xi^\star_{\text{MWR}}$
has a higher probability than random guessing, in choosing a dispatch
which may result in an optimal schedule. This is especially true towards
the end of the schedule building process. Similarly,
$\xi^\star_{\text{LWR}}$ chooses dispatches resulting in optimal
schedules with a higher probability. This would appear to be support the
idea that the higher the probability of dispatching jobs that may lead
to an optimal schedule, the better the SDRs performance, as illustrated
by . However, there is a counter example, $\xi^\star_{\text{SPT}}$ has a
higher probability than random dispatching of selecting a jobs that may
lead to an optimal solution. Nevertheless, the random dispatching
performs better than SPT on problem instances
$\mathcal{P}_{j.rnd}^{10 \times 10}$ and
$\mathcal{P}_{j.rndn}^{10 \times 10}$.

![Probability of SDR being optimal,
$\xi^\star_{\langle\text{SDR}\rangle}$](figures/{trdat_prob_moveIsOptimal_10x10_SDR_xistar}.pdf "fig:")
[fig:opt:SDR:xistar]

Looking at , then $\mathcal{P}_{j.rnd}^{10 \times 10}$ has a relatively
high probability ($70\%$ and above) of choosing an optimal job at
random. However, it is imperative to keep making optimal decisions,
because the consequences of making suboptimal dispatches are unknown. To
demonstrate this depicts mean worst and best case scenario of the
resulting deviation from optimality, $\rho$, once off the optimal track,
defined as follows:

[eq:bwc:opt]

$$\begin{aligned}
\quad \zeta_{\min}^{\star}(k) &:=& \mathbb{E}_{\pi_\star}\left\{
\min_{J_j\in\mathcal{L}^{(k)}}(\rho) \;:\;
\forall C_{\max}^{\bm \chi^j} \gneq C_{\max}^{\pi_\star} \right\} \\
\quad \zeta_{\max}^{\star}(k) &:=& \mathbb{E}_{\pi_\star}\left\{
\max_{J_j\in\mathcal{L}^{(k)}}(\rho) \;:\;
\forall C_{\max}^{\bm \chi^j} \gneq C_{\max}^{\pi_\star} \right\}\end{aligned}$$

Note, that this is given that there is only made one non-optimal
dispatch. Generally, there will be more, and then the compound effects
of making suboptimal decisions cumulate.

It is interesting to observe that for
$\mathcal{P}_{j.rnd}^{10 \times 10}$ and
$\mathcal{P}_{j.rndn}^{10 \times 10}$ making suboptimal decisions later
impacts on the resulting makespan more than doing a mistake early. The
opposite seems to be the case for $\mathcal{P}_{f.rnd}^{10 \times 10}$.
In this case it is imperative to make good decisions right from the
start. This is due to the major structural differences between JSP and
FSP, namely the latter having a homogeneous machine ordering,
constricting the solution immensely.

![Mean deviation from optimality, $\rho$, (%), for best and worst case
scenario of making one suboptimal dispatch (i.e. $\zeta^{\star}_{\min}$
and $\zeta^{\star}_{\max}$), depicted as lower and upper bound,
respectively, for $\mathcal{P}_{j.rnd}^{10 \times 10}$,
$\mathcal{P}_{j.rndn}^{10 \times 10}$ and
$\mathcal{P}_{f.rnd}^{10 \times 10}$. Moreover, mean suboptimal move is
given as a dashed
line.](figures/{stepwise_10x10_OPT_casescenario}.pdf "fig:") [fig:case]

Blended dispatching rules {#sec:opt:bdr}
-------------------------

A naive approach to create a simple blended dispatching rule (BDR) would
be to switch between SDRs at a predetermined time. Observing again , a
presumably good BDR for $\mathcal{P}_{j.rnd}^{10 \times 10}$ would be to
start with $\xi^\star_{\text{SPT}}$ and then switch over to
$\xi^\star_{\text{MWR}}$ at around time step $k=40$, where the SDRs
change places in outperforming one another. A box-plot for $\rho$ for
the BDR compared with MWR and SPT is depicted in and its main statistics
are reported in . This simple swap between SDRs does outperform the SPT
heuristic, yet doesn’t manage to gain the performance edge of MWR. Using
SPT downgrades the performance of MWR. A reason for this lack of
performance of our proposed BDR is perhaps that by starting out with SPT
in the beginning, it sets up the schedules in such a way that it’s quite
greedy and only takes into consideration jobs with shortest immediate
processing times. Now, even though it is possible to find optimal
schedules from this scenario, as shows, the inherent structure that’s
already taking place might make it hard to come across by simple
methods. Therefore, it’s by no means guaranteed that by simply swapping
over to MWR will handle that situation which applying SPT has already
created. does however show, that by applying MWR instead of SPT in the
latter stages, does help the schedule to be more compact w.r.t. SPT.
However, the fact remains that the schedules have diverged too far from
what MWR would have been able to achieve on its own.

![Box-plot for deviation from optimality, $\rho$, (%) for BDR where SPT
is applied for the first 10%, 15%, 20%, 30% or 40% of the dispatches,
followed by MWR](figures/j_rnd/{boxplotRho_BDR_10x10}.pdf "fig:")
[fig:boxplot:BDR]

[t] [tbl:BDR:stats]

<span>ccrlrrrrrr</span> SDR \#1 & SDR \#2 & $k$ & Set & Min. & 1st Qu. &
Median & Mean & 3rd Qu. & Max.\
SPT & – & $K$ & train & 20.38 & 41.15 & 50.70 & 51.31 & 59.18 & 94.20\
SPT & – & $K$ & test & 22.75 & 41.39 & 49.53 & 50.52 & 58.60 & 93.03\
MWR & – & $K$ & train & **4.42** & **17.84** & **21.74** & 22.13 & 26.00
& 47.78\
MWR & – & $K$ & test & **3.37** & **17.07** & 21.39 & 21.65 & 25.98 &
**41.80**\
SPT & MWR & 10 & train & 5.54 & 17.98 & 21.75 & **21.99** & **25.43** &
**44.02**\
SPT & MWR & 10 & test & 5.87 & 17.29 & **20.78** & **21.28** & **24.67**
& 44.47\
SPT & MWR & 15 & train & 4.76 & 18.24 & 22.04 & 22.49 & 26.65 & 49.86\
SPT & MWR & 15 & test & 7.42 & 17.60 & 21.38 & 21.83 & 25.45 & 45.98\
SPT & MWR & 20 & train & 5.76 & 18.98 & 22.46 & 23.01 & 26.97 & 41.59\
SPT & MWR & 20 & test & 8.31 & 18.64 & 22.92 & 23.29 & 27.10 & 49.93\
SPT & MWR & 30 & train & 9.77 & 20.89 & 25.60 & 25.76 & 30.01 & 50.94\
SPT & MWR & 30 & test & 4.39 & 21.20 & 26.08 & 26.25 & 30.58 & 49.88\
SPT & MWR & 40 & train & 13.04 & 23.42 & 28.12 & 28.94 & 33.67 & 54.98\
SPT & MWR & 40 & test & 8.55 & 24.20 & 28.16 & 28.98 & 33.20 & 57.21\

In the stepwise optimality was inspected, given that all committed
dispatches were based on the optimal trajectory. As mistakes are bound
to be made at some points, it is interesting to see how the stepwise
optimality evolves for its intended trajectory, thereby updating to:

$$\quad \label{eq:tracc:track}
\xi_{\pi} := \mathbb{E}_{\pi}\left\{\pi_{\star} = \pi \right\}$$

shows the log likelihood for $\xi_{\langle 
\text{SDR} \rangle}$ using $\mathcal{P}_{j.rnd}^{10 \times 10}$. There
one can see that even though $\xi_{\text{SPT}}$ is generally more likely
to find optimal dispatches in the initial steps, then shortly after
$k=15$, $\xi_{\text{MWR}}$ becomes a contender again. This could explain
why our BDR switch at $k=40$ from was unsuccessful. However, changing to
MWR at $k\leq20$ is not statistically significant from MWR (boost in
mean $\rho$ is at most -0.5%). But as pointed out for , it’s not so
fatal to make bad moves in the very first dispatches for
$\mathcal{P}_{j.rnd}^{10 \times 10}$, hence little gain with improved
classification accuracy in that region. However, after $k>20$ then the
BDR performance starts diverging from that of MWR.

![Log likelihood of SDR being optimal for
$\mathcal{P}_{j.rnd}^{10 \times 10}$, when following its corresponding
SDR trajectory, i.e.,
$\log\left(\xi_{\langle\text{SDR}\rangle}\right)$](figures/j_rnd/{trdat_prob_moveIsOptimal_10x10_SDR_xi}.pdf "fig:")
[fig:opt:SDR:xi]

Preference Learning {#ch:expr:CDR}
===================

demonstrated there is something to be gained by trying out different
combinations of DRs, however, it is non-trivial. In this section one
approach to learning how such combinations is presented. Learning models
considered in this study are based on ordinal regression in which the
learning task is formulated as learning preferences. In the case of
scheduling, learning which operations are preferred to others. Ordinal
regression has been previously presented in @Ru06:PPSN and in @InRu11a
for JSP, and given here for completeness.

The optimum makespan is known for each problem instance. At each time
step $k$, a number of feature pairs are created. Let
$\bm{\phi}^{o}\in\mathcal{F}$ denote the post-decision state when
dispatching $J_o\in\mathcal{O}^{(k)}$ corresponds to an optimal schedule
being built. All post-decisions states corresponding to suboptimal
dispatches, $J_s\in\mathcal{S}^{(k)}$, are denoted by
$\bm{\phi}^{s}\in\mathcal{F}$. Note, , and .

The approach taken here is to verify analytically, at each time step, by
fixing the current temporal schedule as an initial state, whether it is
possible to somehow yield an optimal schedule by manipulating the
remainder of the sequence. This also takes care of the scenario that
having dispatched a job resulting in a different temporal makespan would
have resulted in the same final makespan if another optimal dispatching
sequence would have been chosen. That is to say the training data
generation takes into consideration when there are multiple optimal
solutions[^2] to the same problem instance.

Let’s label features from that were considered optimal, , and
suboptimal, by $y_o=+1$ and $y_s=-1$ respectively. Then, the preference
learning problem is specified by a set of preference pairs:

$$\begin{aligned}
\quad \Psi &=& 
\left\{\left(\bm \psi^o,+1\right),\left(\bm \psi^s,-1\right)
\;:\;
\forall \left(J_o,J_s\right) \in \mathcal{O}^{(k)} \times 
\mathcal{S}^{(k)}\right\}_{k=1}^{K} \nonumber
\\ &\subset& \Phi\times Y \label{eq:prefset}\end{aligned}$$

where $\Phi\subset \mathbb{R}^d$ is the training set of $d=16$ features
(cf. ), $Y=\{+1,-1\}$ is the outcome space from job pairs
$J_o\in\mathcal{O}^{(k)}$ and $J_s\in\mathcal{S}^{(k)}$, for all
dispatch steps $k$.

To summarise, each job is compared against another job of the job-list,
$\mathcal{L}^{(k)}$, and if the makespan differs (i.e
$C_{\max}^{\pi_\star(\bm \chi^s)} \gneq C_{\max}^{\pi_\star(\bm \chi^o)}$)
an optimal/suboptimal pair is created. However, if the makespans are
identical the pair is omitted since they give the same optimal makespan.
This way, only features from a dispatch resulting in a suboptimal
solution is labelled undesirable.

Now let’s consider the model space
$\mathcal{H} = \{\pi(\cdot) : X \mapsto Y\}$ of mappings from solutions
to ranks. Each such function $\pi$ induces an ordering on the solutions
by the following rule:

$$\quad\label{eq:linear}
\bm \chi^i \succ \bm \chi^j \quad \Leftrightarrow \quad \pi(\bm \chi^i) > 
\pi(\bm \chi^j)$$

where the symbol $\succ$ denotes “is preferred to.” The function used to
induce the preference is defined by a linear function in the feature
space:

$$\quad 
\pi(\bm \chi^j)=\sum_{i=1}^d w_i\phi_i(\bm \chi^j)=\big<{\vec{w}}\cdot{\bm{\phi}(\bm \chi^j)}\big>.$$

Logistic regression learns the optimal parameters
$\vec{w}^*\in\mathbb{R}^d$. For this study, L2-regularised logistic
regression from the liblinear package @liblinear without bias is used to
learn the preference set $\Psi$, defined by . Hence the job chosen to be
dispatched, $J_{j^*}$, is the one corresponding to the highest
preference estimate, i.e., where $\pi(\cdot)$ is the classification
model obtained by the preference set.

Preliminary experiments for creating step-by-step model was done in
@InRu11a resulting in local linear model for each dispatch; a total of
$K$ linear models for solving $n\times m$ JSP. However, the experiments
there showed that by fixing the weights to its mean value throughout the
dispatching sequence results remained satisfactory. A more sophisticated
way would be to create a new linear model, where the preference set,
$\Psi$, is the union of the preference pairs across the $K$ dispatches,
such as described in . This would amount to a substantial preference
set, and for $\Psi$ to be computationally feasible to learn, $\Psi$ has
to be reduced. For this several ranking strategies were explored in
@InRu15a, the results there showed that it’s sufficient to use partial
subsequent rankings, namely, combinations of $r_i$ and $r_{i+1}$ for
$i\in\{1,\ldots,n'\}$, are added to the preference set, where
$r_1>r_2>\ldots>r_{n'}$ ($n'\leq n$) are the rankings of the job-list,
in such a manner that in the cases that there are more than one
operation with the same ranking, only one from that rank is needed to be
compared to the subsequent rank. Moreover, for this study, which deals
with $10\times 10$ problem instances instead of $6\times5$, the partial
subsequent ranking becomes necessary, as full ranking is computationally
infeasible due to its size. Defining the size of the preference set as
$l=\lvert\Psi\rvert$, then if $l$ is too large re-sampling to size
$l_{\max}$ may be needed to be done in order for the ordinal regression
to be computationally feasible.

The training data from @InRu11a was created from optimal solutions of
randomly generated problem instances, i.e., traditional *passive
imitation learning* (PIL). As JSP is a sequential decision making
process, errors are bound to emerge. Due to compound effect of making
suboptimal dispatches, the model leads the schedule astray from learned
feature-space, resulting in the new input being foreign to the learned
model. Alternatively, training data could be generated using suboptimal
solution trajectories as well, as was done in @InRu15a, where the
training data also incorporated following the trajectories obtained by
applying successful SDRs from the literature. The reasoning behind it
was that they would be beneficial for learning, as they might help the
model to escape from local minima once off the coveted optimal path.
Simply aggregating training data obtained by following the trajectories
of well-known SDRs yielded better models with lower deviation from
optimality, $\rho$.

Inspired by the work of @RossB10 [@RossGB11], the methodology of
generating training data will now be such that it will iteratively
improve upon the model, such that the feature-space learned will be
representative of the feature-space the eventual model would likely
encounter, known as DAgger for *active imitation learning* (AIL).
Thereby, eliminating the ad-hoc nature of choosing trajectories to
learn, by rather letting the model lead its own way in a
self-perpetuating manner until it converges.

Furthermore, in order to boost training accuracy, two strategies were
explored

1.  [expr:boost:varylmax] increasing number of preferences used in
    training (i.e. varying ),

2.  [expr:boost:newdata] introducing more problem instances (denoted EXT
    in experimental setting).

Note, the following experimental studies will address
[expr:boost:newdata], whereas preliminary experiments for
[expr:boost:varylmax] showed no statistical significance in boost of
performance. Hence, the default set-up will be $l_{\max}=5 \cdot 10^5$
which is roughly the amount of features encountered from one pass of
sampling a trajectory using a fixed policy $\pi$ for the default
$N_{\text{train}}=300$.

Another way to adjust training accuracy is to give different weight to
various time steps. To address this problem, two different stepwise
sampling biases (or data balancing techniques) will be considered

1.  [bias:equal] **(equal)** where each time step has equal probability,
    this was used in @InRu14 [@InRu15a] and serves as a baseline.

2.  [bias:adjdbl2nd] **(adjdbl2nd)** where each time step is adjusted to
    the number of preference pairs for that particular step (i.e. each
    step now has equal probability irrespective of quantity of
    encountered features). This is done with re-sampling. In addition,
    there is superimposed twice as much likelihood of choosing pairs
    from the latter half of the dispatching process. Then the final
    sampled data set is divided as follows:
    $\lvert\{\Psi(k)\}_{k=0}^{\frac{K}{2}-1}\rvert \approx \frac{1}{3}l_{\max}$
    and
    $\lvert\{\Psi(k)\}_{k=\frac{K}{2}}^{K-1}\rvert \approx \frac{2}{3}l_{\max}$.

Remark: as the following s require repeated collection of training data,
and since its labelling is a very time intensive task the remainder of
the paper will solely be focusing on
$\mathcal{P}_{j.rnd}^{10 \times 10}$.

Passive Imitation Learning {#sec:il:passive}
==========================

Using the terms from game-theory used in @CesaBianchi06, then our
problem is a basic version of the sequential prediction problem where
the predictor (or forecaster), $\pi$, observes each element of a
sequence $\bm \chi$ of jobs, where at each time step
$k \in \{1,...,K\}$, before the $k$-th job of the sequence is revealed,
the predictor guesses its value $\chi_k$ on the basis of the previous
$k-1$ observations.

Prediction with Expert Advice {#sec:expertPolicy}
-----------------------------

Let us assume one knows the expert policy $\pi^\star$, which can query
what is the optimal choice of $\chi_k={j^*}$ at any given time step $k$.
Now let’s use to back-propagate the relationship between post-decision
states and $\hat{\pi}$ with preference learning via our collected
feature set, denoted $\Phi^\text{OPT}$, i.e., collecting the features
set corresponding following optimal tasks $J_{j^*}$ from $\pi^\star$ in
. This baseline sampling trajectory originally introduced in @InRu11a
for adding features to the feature set is a pure strategy where at each
dispatch an optimal task is dispatched.

By querying the expert policy, $\pi_\star$, the ranking of the job-list,
$\mathcal{L}$, is determined such that:

$$\quad
r_1 \succ r_2 \succ \cdots \succ r_{n'} \quad (n' \leq n)$$

implies $r_1$ is preferable to $r_2$, and $r_2$ is preferable to $r_3$,
etc. In this study, then it’s known that
$r \propto C_{\max}^{\pi_\star}$, hence the optimal job-list is the
following:

$$\quad
\mathcal{O}=\left\{r_i \;:\; r_i \propto \min_{J_j \in \mathcal{L}} 
C_{\max}^{\pi_\star(\bm \chi^j)}\right\}$$

found by solving the current partial schedule to optimality using a MIP
solver.

When $\lvert\mathcal{O}^{(k)}\rvert>1$, there can be several
trajectories worth exploring. However, only one is chosen at random.
This is deemed sufficient as the number of problem instances,
$N_{\text{train}}$, is relatively large.

Follow the Perturbed Leader {#sec:perturbedLeader}
---------------------------

By allowing a predictor to randomise it’s possible to achieve improved
performance @CesaBianchi06 [@Hannan57]. This is the inspiration for our
next strategy called Follow the Perturbed Leader, denoted OPT$\epsilon$.
Its pseudo code is given in and describes how the expert policy (i.e.
optimal trajectory) from is subtly “perturbed” with $\epsilon=10\%$
likelihood, by choosing a job corresponding to the second best
$C_{\max}$ instead of a optimal one with some small probability.

[t] [pseudo:perturbedLeader]

[1] Ranking $r_1 \succ r_2 \succ \cdots > r_{n'} ~ (n' \leq n)$ of
$\mathcal{L}$ $\epsilon \gets 0.1$ $p \gets \mathcal{U}(0,1)\in [0,1]$
$\mathcal{O} \gets \left\{j\in\mathcal{L}\;:\;r_j=r_1\right\}$
$\mathcal{S} \gets \left\{j\in\mathcal{L}\;:\;r_j>r_1\right\}$
$j^* \in \left\{j\in\mathcal{S}\;:\;r_j=r_2\right\}$
$j^* \in\mathcal{O}$

Experimental study {#sec:pil:expr}
------------------

![Box plot for $\mathcal{P}_{j.rnd}^{10 \times 10}$ deviation from
optimality, $\rho$, using either expert policy and following perturbed
leader.](figures/{j_rnd}/{boxplot_passive_10x10}.pdf "fig:")[fig:passive:boxplot]

Results for using $\mathcal{P}_{j.rnd}^{10 \times 10}$ box-plot of
deviation from optimality, $\rho$, is given in and main statistics are
reported in . To address [expr:boost:newdata], the extended training set
was simply obtained by iterating over more examples, namely
$N^{\text{OPT}}_{\text{train, 
EXT}}=1000$. However, one can see that the increased number of varied
features dissuades the preference models to achieving a good performance
w.r.t. $\rho$. It’s preferable to use the default
$N^{\text{OPT}}_{\text{train}}=300$ and allowing slight perturbations of
the optimal trajectory, as done for $\Phi^{\text{OPT}\epsilon}$.
Unfortunately, all this overhead has not managed to surpass MWR in
performance, except for $\Phi^{\text{OPT}\epsilon}$ using
[bias:adjdbl2nd] with a $\Delta\rho\approx-4.24\%$ boost in mean
performance. Otherwise, for [bias:equal], there is a loss of
$\Delta\rho\approx+6.23\%$ in mean performance. This is likely due to
the fact that if equal probability is used for stepwise sampling, then
there are hardly any emphasis given to the final dispatches as there a
relatively few (compared to previous steps) preference pairs belonging
to those final stages. Revisiting , then the band for
$\{\zeta^{\star}_{\min},\zeta^{\star}_{\max}\}$ is quite tight, as the
problem is immensely constricted and few operations to choose from.
However, the empirical evidence from using [bias:adjdbl2nd] shows that
it is imperative to make right decisions at the very end.

Based on the results from @InRu11a the expert policy is a promising
starting point. However, that was for $6\times5$ dimensionality (i.e.
$K=30$), which is a much simpler problem space. Notice that in there was
virtually no chance for $\xi_\pi(k)$ of choosing a job resulting in
optimal makespan after step $k=28$. Since job-shop is a sequential
prediction problem, all future observations are dependent on previous
operations. Therefore, learning sampled features that correspond only to
optimal or near-optimal schedules isn’t of much use when the preference
model has diverged too far. showed that good classification accuracy
based on $\xi^\star_\pi$ does not necessarily mean a low mean deviation
from optimality, $\rho$. This is due to the learner’s predictions
affects future input observations during its execution, which violates
the crucial i.i.d. assumptions of the learning approach, and ignoring
this interaction leads to poor performance. In fact, @RossB10 proves
that assuming the preference model has a training error of $\epsilon$,
then the total compound error (for all $K$ dispatches) the classifier
induces itself grows quadratically, ${O}\left(\epsilon 
K^2\right)$, for the entire schedule, rather than having linear loss,
${O}\left(\epsilon K\right)$, if it were i.i.d.

Active Imitation Learning {#sec:il:active}
=========================

To amend performance from $\Phi^{\text{OPT}}$-based models, suboptimal
partial schedules were explored in @InRu15a by inspecting the features
from successful SDRs, $\Phi^{\langle\text{SDR}\rangle}$, by passively
observing a full execution of following the task chosen by the
corresponding SDR. This required some trial-and-error as the experiments
showed that features obtained by SDR trajectories were not equally
useful for learning.

To automate this process, inspiration from AIL presented in @RossGB11 is
sought, called *Dataset Aggregation* (DAgger) method, which addresses a
no-regret algorithm in an on-line learning setting. The novel
meta-algorithm for IL learns a deterministic policy guaranteed to
perform well under its induced distribution of states. The method is
closely related to Follow-the-leader (cf. ), however, with a more
sophisticated leverage to the expert policy. In short, it entails the
model $\pi_i$ that queries an expert policy (same as in ), $\pi_\star$,
it’s trying to mimic, but also ensuring the learned model updates itself
in an iterative fashion, until it converges. The benefit of this
approach is that the feature-states that are likely to occur in practice
are also investigated and as such used to dissuade the model from making
poor choices. In fact, the method queries the expert about the desired
action at individual post-decision states which are both based on past
queries, and the learner’s interaction with the current environment.

DAgger has been proven successful on a variety of benchmarks @RossGB11
[@Ross13], such as the video games Super Tux Kart and Super Mario Bros.,
handwriting recognition and autonomous navigation for large unmanned
aerial vehicles. In all cases greatly improving traditional supervised
IL approaches.

DAgger
------

The policy of AIL at iteration $i>0$ is a mixed strategy given as
follows:

$$\quad\label{eq:il}
\pi_i = \beta_i\pi_\star + (1-\beta_i)\hat{\pi}_{i-1}$$

where $\pi_\star$ is the expert policy and $\hat{\pi}_{i-1}$ is the
learned model from the previous iteration. Note, for the initial
iteration, $i=0$, a pure strategy of $\pi_\star$ is followed. Hence,
$\hat{\pi}_0$ corresponds to the preference model from (i.e.
$\Phi^{\text{IL}0}=\Phi^{\text{OPT}}$).

shows that $\beta_i$ controls the probability distribution of querying
the expert policy $\pi_\star$ instead of the previous imitation model,
$\hat{\pi}_{i-1}$. The only requirement for $\{\beta_i\}_i^\infty$
according to @RossGB11 is that
$\lim_{T\to\infty}\frac{1}{T}\sum_{i=0}^T\beta_i=0$ to guarantee finding
a policy $\hat{\pi}_i$ that achieves $\epsilon$ surrogate loss under its
own state distribution limit.

explains the pseudo code for how to collect partial training set,
$\Phi^{\text{IL}i}$ for $i$-th iteration of AIL. Subsequently, the
resulting preference model, $\hat{\pi}_i$, learns on the aggregated
datasets from all previous iterations, namely:

$$\quad\label{eq:DAgger}
\Phi^{\text{DA}i}=\bigcup_{i'=0}^{i}\Phi^{\text{IL}i'}$$

and its update procedure is detailed in .

[t] [pseudo:activeIL]

[1] $i\geq0$ Ranking $r_1 \succ r_2 \succ \cdots > r_{n'} ~ (n' \leq n)$
of $\mathcal{L}$ $p \gets \mathcal{U}(0,1)\in [0,1]$   (unsupervised)
$\beta_i \gets 0$   (fixed supervision) $\beta_i \gets 1$
$j^* \gets \mathop{\rm argmax}_{j\in 
        \mathcal{L}}\{I_j^{\hat{\pi}_{i-1}}\}$
$\mathcal{O} \gets \left\{j\in\mathcal{L}\;:\;r_j=r_1\right\}$
$j^* \in\mathcal{O}$

[t] [pseudo:DAgger]

[1] $T\geq1$ $\Phi^{\text{IL}0} \gets \Phi^{\text{OPT}}$
$\hat{\pi}_0 \gets$ Let
$\pi_i = \beta_i\pi_\star + (1-\beta_i)\hat{\pi}_{i-1}$ Sample $K$-step
tracks using $\pi_i$ $\Phi^{\text{IL}i} = \{(s,\pi_\star(s))\}$
$\Phi^{\text{DA}i} \gets \Phi^{\text{DA}i-1} \cup 
        \Phi^{\text{IL}i}$ $\hat{\pi}_{i+1} \gets$ best $\hat{\pi}_i$ on
validation

Results {#sec:ail:expr}
-------

Due to time constraints, only $T=3$ iterations will be inspected. In
addition, preliminary experiments using DAgger for JSP favoured a simple
parameter-free version of $\beta_i$ in . Namely, the mixed strategy for
$\{\beta_i\}_{i=0}^T$ is unsupervised with $\beta_i=I(i=0)$, where $I$
is the indicator function.[^3]

Regarding [expr:boost:newdata] strategy, showed that adding new problem
instances did not boost performance for the expert policy (which is
equivalent for the initial iteration of DAgger). Hence, for active IL,
the extended set is now consists of each iteration encountering
$N_{\text{train}}$ new problem instances. For a grand total of:

$$\quad
N^{\text{DA}i}_{\text{train, EXT}}=N_{\text{train}}\cdot (i+1)$$

problem instances explored for the aggregated extended training set used
for the learning model at iteration $i$. This way, the extended training
data is used sparingly, as labelling for each problem instances is
computationally intensive. As a result, the computational budget for
DAgger is same regardless whether there are new problem instances used
or not, i.e.,
$\lvert\Phi^{\text{DA}i}\rvert\approx\lvert\Phi^{\text{DA}i}_{\text{EXT}}\rvert$.

Results for $\mathcal{P}_{j.rnd}^{10 \times 10}$ box-plot of deviation
from optimality, $\rho$, is given in and main statistics are reported in
. As one can see, DAgger is not fruitful when the same problem instances
are continually used. This is due to the fact that there is not enough
variance between $\Phi^{\text{IL}i}$ and $\Phi^{\text{IL}(i-1)}$, hence
the aggregated feature set $\Phi^{\text{DA}i}$ is only slightly
perturbed with each iterations. Which from showed it was not a very
successful modification for the expert policy. Although, it’s noted that
by introducing suboptimal feature-space the preference model is not as
drastically bad as the extended optimal policy, even though
$\lvert\Phi^{\text{DA}i}\rvert\approx\lvert\Phi^{\text{OPT}}_{\text{EXT}}\rvert$.
However, when using new problem instances at each iterations, the
feature set becomes varied enough that situations arise that can be
learned to achieve a better represented classification problem which
yields a lower mean deviation from optimality, $\rho$.

![Box plot for $\mathcal{P}_{j.rnd}^{10 \times 10}$ deviation from
optimality, $\rho$, using DAgger for
JSP](figures/{j_rnd}/{boxplot_active_10x10}.pdf "fig:")
[fig:active:boxplot]

Summary of Imitation Learning {#sec:il:expr}
=============================

A summary of $\mathcal{P}_{j.rnd}^{10 \times 10}$ best PIL and AIL
models w.r.t. deviation from optimality, $\rho$, from , respectively,
are illustrated in , and main statistics are given in . To summarise,
the following trajectories were used

expert policy, trained on $\Phi^{\text{OPT}}$

perturbed leader, trained on $\Phi^{\text{OPT}\epsilon}$

imitation learning, trained on $\Phi^{\text{DA}i}_{\text{EXT}}$ for
iterations $i=\{1,\ldots,3\}$ using extended training set

As a reference, the single priority dispatching rule MWR is shown at the
edges of .

At first one can see that the perturbed leader ever so-slightly improves
the mean for $\rho$, rather than using the baseline expert policy.
However, AIL is by far the best improvement. With each iteration of
DAgger, the models improve upon the previous iteration

for [bias:equal] with [expr:boost:newdata] then $i=1$ starts with
increasing $\Delta\rho\approx+1.39\%$. However, after that first
iteration there is a performance boost of $\Delta\rho\approx-15.11\%$
after $i=2$ and $\Delta\rho\approx-0.19\%$ for the final iteration $i=3$

on the other hand when using [bias:adjdbl2nd] with [expr:boost:newdata],
only one iteration is needed, as $\Delta\rho\approx-11.68$ for $i=1$,
and after that it stagnates with $\Delta\rho\approx+0.55\%$ for $i=2$
and for $i=3$ it is significantly worse than the previous iteration by
$\Delta\rho\approx+0.75\%$

In both cases, DAgger outperforms MWR

after $i=3$ iterations by $\Delta\rho\approx-5.31\%$ for [bias:equal]
with [expr:boost:newdata]

after $i=1$ iteration by $\Delta\rho\approx-9.31\%$ for [bias:adjdbl2nd]
with [expr:boost:newdata]

Note, for [bias:equal] without [expr:boost:newdata], then DAgger is
unsuccessful, and the aggregated data set downgrades the performance of
the previous iterations, making it best to learn solely on the initial
expert policy for that model configuration.

Regarding [expr:boost:newdata], then it’s not successful for the expert
policy, as $\rho$ increased approximately 10%. This could most likely be
counter-acted by increasing $l_{\max}$ to reflect the 700 additional
examples. What is interesting though, is that [expr:boost:newdata] is
well suited for AIL, using the same $l_{\max}$ as before. Note, the
amount of problems used for $N^{\text{OPT}}_{\text{train, EXT}}$ is
equivalent to $T=2\tfrac{1}{3}$ iterations of extended DAgger. The new
varied data gives the aggregated feature set more information of what is
important to learn in subsequent iterations, as those new feature-states
are more likely to be encountered ‘in practice.’ Not only does the AIL
converge faster, it also consistently improves with each iterations.

![Box plot for $\mathcal{P}_{j.rnd}^{10 \times 10}$ deviation from
optimality, $\rho$, using either expert policy, DAgger or following
perturbed leader strategies. MWR shown for
reference.](figures/{j_rnd}/{boxplot_summary_10x10}.pdf "fig:")[fig:all:boxplot]

[t] [tbl:IL:stats]

<span>c@rrrrrrrrrr</span> $\pi$[^4] & $T$[^5] & Bias & Set &
$N_{\text{train}}$ & Min. & 1st Qu. & Median & Mean & 3rd Qu. & Max.\
OPT & 0 & adjdbl2nd & train & 300 & 6.05 & 18.60 & 23.85 & 24.50 & 29.04
& 55.81\
OPT & 0 & adjdbl2nd & test & 300 & 5.56 & 19.16 & 24.24 & 25.19 & 30.42
& 55.52\
OPT & 0 & equal & train & 300 & 7.87 & 23.34 & 29.30 & 30.73 & 36.47 &
61.45\
OPT & 0 & equal & test & 300 & 8.31 & 23.88 & 30.32 & 31.46 & 37.70 &
67.24\
DA1 & 1 & adjdbl2nd & train & 600 & 2.08 & **9.44** & **12.30** &
**12.82** & **15.67** & **29.63**\
DA1 & 1 & adjdbl2nd & test & 300 & **0.00** & **9.22** & **12.39** &
**12.73** & **15.85** & 35.17\
DA1 & 1 & equal & train & 600 & 9.47 & 24.92 & 31.51 & 32.12 & 37.96 &
66.29\
DA1 & 1 & equal & test & 300 & 4.77 & 23.77 & 30.34 & 31.40 & 37.81 &
73.73\
DA2 & 2 & adjdbl2nd & train & 900 & **0.93** & 10.01 & 12.91 & 13.37 &
16.40 & 31.19\
DA2 & 2 & adjdbl2nd & test & 300 & 0.39 & 9.84 & 13.13 & 13.44 & 16.62 &
**34.57**\
DA2 & 2 & equal & train & 900 & 2.36 & 12.82 & 16.65 & 17.01 & 21.06 &
39.25\
DA2 & 2 & equal & test & 300 & 1.72 & 12.57 & 16.38 & 16.89 & 20.66 &
42.44\
DA3 & 3 & adjdbl2nd & train & 1200 & 0.93 & 10.45 & 13.71 & 14.12 &
17.15 & 32.91\
DA3 & 3 & adjdbl2nd & test & 300 & 0.87 & 10.44 & 13.64 & 14.08 & 17.23
& 34.41\
DA3 & 3 & equal & train & 1200 & 0.98 & 12.50 & 16.28 & 16.82 & 20.67 &
37.93\
DA3 & 3 & equal & test & 300 & 0.26 & 12.32 & 16.01 & 16.52 & 20.22 &
41.62\
OPT$\epsilon$ & 0 & adjdbl2nd & train & 300 & 4.64 & 13.63 & 17.56 &
18.07 & 21.66 & 36.25\
OPT$\epsilon$ & 0 & adjdbl2nd & test & 300 & 1.91 & 13.18 & 16.48 &
16.89 & 20.28 & 35.60\
OPT$\epsilon$ & 0 & equal & train & 300 & 4.52 & 21.31 & 27.63 & 28.04 &
33.69 & 63.74\
OPT$\epsilon$ & 0 & equal & test & 300 & 8.54 & 22.03 & 27.26 & 27.94 &
33.02 & 60.38\

Discussion and conclusions {#sec:con}
==========================

The single priority dispatching rules remain a popular approach to
scheduling, as they are simple to implement and quite efficient.
Nevertheless, when they are successful and when they fail remains
illusive. By inspecting optimal schedules, and investigating the
probability that an optimal dispatch could be chosen by chance, and by
looking at the impact of choosing sub-optimal dispatches, some light is
shed on how SDRs vary in performance. Furthermore, the problem instance
space was varied, giving an even better understanding of the behaviour
of the SDRs. This analysis, however, also revealed that creating new
dispatching rules from data is by no means trivial.

Experiments in show that following the optimal policy is not without its
faults. There are many obstacles to consider in order to improve model
configurations. When training the learning model, there is a trade-off
between making the over-all best decisions (in terms of highest mean
validation accuracy) versus making the right decision on crucial time
points in the scheduling process, as clearly illustrated. Moreover,
before training the learned model, the preference set $\Psi$ needs to be
re-sampled to size $l_{\max}$. As the effects of making suboptimal
choices varies as a function of time, the stepwise bias should rather
take into account the disproportional amount of features during the
dispatching process. As the experimental studies in showed, instead of
equal probability (i.e. [bias:equal]) it was much more fruitful to
adjust the set to its number of preference and doubling the emphasis on
the second half (i.e. [bias:adjdbl2nd]). However, there are many other
stepwise sampling strategies based on our analysis that could have been
chosen instead, as here only a simplification of the trend from was
chosen. This also opens up the question of how should validation
accuracy be measured? Since the model is based on learning preferences,
both based on optimal versus suboptimal, and then varying degrees of
sub-optimality. Since ranks are only looked at in a black and white
fashion, such that the makespans need to be strictly greater to belong
to a higher rank, then it can be argued that some ranks should be
grouped together if their makespans are sufficiently close. This would
simplify the training set, making it (presumably) have less
contradictions and be more appropriate for linear learning. Or simply
the validation accuracy could be weighted w.r.t. the difference in
makespan. During the dispatching process, there are some significant
time points which need to be especially taken care off. showed how
making suboptimal decisions is especially critical during the later
stages for job-shop, whereas for flow-shop the earlier stages of
dispatches are more critical.

Despite the information gathered by following an optimal trajectory, the
knowledge obtained is not enough by itself. Since the learning model
isn’t perfect, it is bound to make a suboptimal dispatch eventually.
When it does, the model is in uncharted territory as there is no
certainty the samples already collected are able to explain the current
situation. For this we propose investigating partial schedules from
suboptimal trajectories as well, since the future observations depend on
previous predictions. A straight forward approach would be to inspect
the trajectories of promising SDRs or CDRs. However, more information is
gained when applying AIL inspired by work of @RossB10 [@RossGB11], such
that the learned policy following an optimal trajectory is used to
collect training data, and the learned model is iteratively updated.
This can be done over several iterations, with the benefit being, that
the scheduling features that are likely to occur in practice are
investigated, and as such used to dissuade the model from making poor
choices in the future.

The main drawback of DAgger is that it quite aggressively queries the
expert, making it impractical for some problems, especially if it
involves human experts. A way to confront that, @Kim13 [@Judah12]
propose frameworks to minimise the expert’s labelling effort. Or even
circumvent the expert policy altogether by using a ‘poorer’ reference
policy instead (i.e. $\pi_\star$ in is suboptimal) @ChangKADL15.

This study has been structured around the job-shop scheduling problem,
however, it can be easily extended to other types of deterministic
optimisation problems that involve sequential decision making. The
framework presented here collects snap-shots of the partial schedules by
following an optimal trajectory, and verifying the resulting optimal
solution from each possible state. From which the stepwise optimality of
individual features can be inspected, and its inference could for
instance justify omittance in feature selection. Moreover, by looking at
the best and worst case scenario of suboptimal dispatches, it is
possible to pinpoint vulnerable times in the scheduling process.

[^1]: Dispatch and time step are used interchangeably.

[^2]: There can be several optimal solutions available for each problem
    instance. However, it is deemed sufficient to inspect only one
    optimal trajectory per problem instance as there are
    $N_{\text{train}}=300$ independent instances which gives the
    training data variety.

[^3]: $\beta_0=1$ and $\beta_i=0,\forall i>0$.

[^4]: For DAgger, then $T=0$ is conventional expert policy (i.e.
    $\text{DA}0=\text{OPT}$).

[^5]: If $T=0$ then *passive* imitation learning. Otherwise, for $T>0$
    it is considered *active* imitation learning.
