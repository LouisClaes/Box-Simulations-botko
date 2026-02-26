# RL MCTS Hybrid - Volledige Workflow Uitleg Voor Expert Review (NL)

## 1. Doel van dit document
Dit document is expliciet bedoeld voor externe inhoudelijke review door een zeer ervaren expert.
Ik schrijf dit alsof ik als student mijn keuzes verantwoord, inclusief:
1. hoe de workflow echt werkt in de code;
2. waarom bepaalde ontwerpkeuzes zijn gemaakt;
3. waar nog zwakke punten of onzekerheden zitten;
4. welke gerichte vragen ik aan de expert wil stellen.

Dit document staat naast de README en is bedoeld als diepere technische bespreking.

## 2. Korte samenvatting in gewone taal
`rl_mcts_hybrid` is een hierarchische RL-strategie voor 3D bin packing:
1. het model encodeert de toestand (bins + box + buffer);
2. high-level policy kiest een actie-context (place/skip/reconsider + box/bin);
3. low-level policy kiest een concrete plaatsingskandidaat;
4. optioneel verfijnt MCTS de keuze;
5. omgeving voert actie uit, reward terug, en PPO update volgt.

Mijn hoofdreden om deze strategie eerst te trainen voor de thesis:
1. dit is de rijkste architectuur (policy + model + search);
2. je kunt sterkere onderzoeksvragen testen dan bij pure DQN/PPO;
3. je krijgt duidelijke ablations (met/zonder MCTS, met/zonder modelkwaliteit).

## 3. End-to-end workflow zoals ik die nu begrijp
### 3.1 Train-entrypoint
`train.py` initialiseert config, model, optimizer, logging, checkpoint-resume en trainingsfasen.

### 3.2 Fase A - Imitation warm-start
Doel:
1. niet vanaf random policy starten;
2. low-level kandidaatselectie sneller op gang krijgen.

Werkwijze:
1. demonstraties van heuristieken;
2. supervised pretrain op acties;
3. checkpoint na imitation.

Keuze:
1. dit verkleint sample-complexity vroeg in training;
2. dit helpt vooral bij grote/ruisige actie-ruimtes.

### 3.3 Fase B - Curriculum RL
Doel:
1. probleemcomplexiteit gecontroleerd opbouwen;
2. eerst simpele placement-logica, daarna full setting.

Typische progression:
1. eenvoudig (single box/single bin);
2. sequencing;
3. multi-bin selectie;
4. volledige setting.

Keuze:
1. curriculum verlaagt kans op instabiel leren;
2. maakt debuggen per stage mogelijk.

### 3.4 Fase C - PPO updates
Per rollout:
1. buffer vullen met transitions;
2. advantages/returns berekenen (GAE);
3. high-level en low-level policy tegelijk updaten;
4. auxiliary losses toepassen.

### 3.5 Evaluatie en checkpoints
1. periodieke eval;
2. best checkpoint bij betere fill;
3. periodieke `latest`/`step_x` checkpoints;
4. final checkpoint bij normale afronding.

### 3.6 HPC orchestration (buiten deze map)
Pipeline in `rl_common/hpc/run_rl_pipeline.py` doet:
1. train;
2. evaluate;
3. visualize;
4. manifest/logs/artifactbeheer.

## 4. Ontwerpkeuzes en motivatie
### 4.1 Hierarchische policy (HL + LL)
Waarom:
1. natuurlijke taakdecompositie: eerst wat/waar globaal, dan exacte placement;
2. minder brute-force dan een vlakke enorme actie-ruimte.

Trade-off:
1. complexere trainingsdynamiek;
2. meer kans op inconsistentie tussen HL-beslissing en LL-kandidaatset.

### 4.2 Candidate generator + pointer policy
Waarom:
1. domeinkennis in candidate generator beperkt onzinnige acties;
2. pointer-netwerk kiest uit kwaliteitssubset in plaats van volledige grid.

Trade-off:
1. prestaties hangen sterk af van candidate kwaliteit/diversiteit;
2. bias in kandidaatconstructie kan policy begrenzen.

### 4.3 World model + MCTS
Waarom:
1. lookahead kan lokale greediness corrigeren;
2. thesismatig interessant om model-based component te evalueren.

Trade-off:
1. MCTS staat of valt met transitionkwaliteit;
2. extra compute en meer moving parts.

## 5. Belangrijk punt: world-model transition supervision is beperkt
Huidige status:
1. world model krijgt vooral supervised signal op reward prediction;
2. plus auxiliary signal op void fraction;
3. maar transition-output (latente volgende toestand / next-state heads) heeft beperkte directe supervisie.

Wat dit praktisch betekent:
1. de planner kan op onbetrouwbare state rollouts plannen;
2. MCTS-kwaliteit kan achterblijven ondanks nette policy-loss;
3. hogere kans op "plausibel ogende maar foutieve" lookahead.

Waarom dit thesisrelevant is:
1. als MCTS geen winst geeft, kan dat liggen aan modelkwaliteit in plaats van conceptuele misfit;
2. negatieve resultaten zonder deze nuance zijn methodologisch zwakker.

### Mijn vragen aan expert hierover
1. Welke minimale transition-targets zijn volgens u noodzakelijk voor geloofwaardige model-based planning in deze setting?
2. Is latente-state supervision voldoende, of moeten we expliciet op fysieke observabelen (heightmaps/contact features) trainen?
3. Wanneer is een eenvoudiger model-free baseline methodologisch sterker dan een half-supervised world model?

## 6. Exploration -> Exploitation: standaard leer-arc
Je vroeg expliciet dat dit standaard moet verlopen: van hoge exploratie naar hoge exploitatie.

### 6.1 Wat nu al aanwezig is
1. PPO samplet stochastisch tijdens training (dus exploratie aanwezig);
2. entropy-term (`ent_coeff`) stimuleert exploratie;
3. curriculum bouwt complexiteit op;
4. imitation-weight wordt geannealed richting 0.

### 6.2 Wat nog niet sterk genoeg expliciet is
1. er is geen harde, expliciete entropy-annealing schedule van hoog naar laag in alle paden;
2. MCTS-train/inference intensiteit is niet volledig als gestandaardiseerde exploratie->exploitatie curve opgezet.

### 6.3 Aanbevolen standaard in dit project
1. Start met hogere entropy-gewichting en lineair/cosine afbouwen.
2. Houd policy sampling in vroege fase stochastisch, later meer greedy.
3. Verhoog MCTS-betrouwbaarheid pas wanneer world-model validatie voldoende is.
4. Rapporteer schedule expliciet in thesis zodat resultaten reproduceerbaar zijn.

### Vragen aan expert
1. Welk annealing-profiel adviseert u hier: lineair, cosine, of adaptief op KL/plateau?
2. Zou u MCTS pas activeren na een meetbare world-model drempel (bijvoorbeeld transition error threshold)?
3. Is exploitatie in uw ervaring beter via entropy-decay of via lagere action-temperature (of beide)?

## 7. Waar physics-simulatie waarschijnlijk waarde toevoegt
### 7.1 Plaatsen waar physics nu te geometrisch kan zijn
1. support-checks zijn vereenvoudigd;
2. contact/wrijving/impact zijn beperkt gemodelleerd;
3. materiaaldeformatie (dooscompressie) ontbreekt grotendeels.

### 7.2 Concreet waar ik physics zou toevoegen
1. Candidate validation: voeg stabiteitscriteria toe (kantelmoment, contactpatch).
2. Reward shaping: straf fysiek instabiele placements zwaarder.
3. World-model targets: voorspel fysisch-relevante indicatoren (stabiliteit, slip-risico).
4. Evaluation suite: aparte physics stress test set.

### Vragen aan expert
1. Welke fysica is must-have voor academisch geloofwaardige claims in dit domein?
2. Waar ligt de juiste balans tussen simulatiefidelity en trainingstijd op HPC?
3. Zijn quasi-static criteria voldoende, of is dynamica (impact/trillingen) noodzakelijk?

## 8. Kritische open punten die ik expliciet wil laten reviewen
1. Is de huidige action-space semantiek (`skip/reconsider/place`) inhoudelijk juist voor operationele doelen?
2. Is de verhouding policy-loss versus auxiliary losses goed gekozen?
3. Is de evaluatieset representatief genoeg voor echte operationele variatie?
4. Is throughput (`ms/box`) zwaar genoeg meegewogen naast fill-rate?
5. Is de vergelijking met andere strategieen fair qua compute-budget en tuning-budget?
6. Is de interpretatie van "MCTS winst" valide als transition-supervision nog beperkt is?

## 9. Wat ik als student nog onzeker vind
1. Of de huidige world-model signaalsterkte genoeg is om MCTS echt betrouwbaar te maken.
2. Of de kandidaatgenerator niet te veel inductieve bias injecteert.
3. Of de huidige rewardmix stabiel generaliseert naar out-of-distribution boxsets.
4. Of physicsvereenvoudigingen te agressief zijn voor sterke conclusies.

## 10. Voorgestelde experimentele volgorde voor thesis
1. Train en valideer `rl_mcts_hybrid` eerst (quick -> full).
2. Draai vaste benchmark op dezelfde seeds/episodes voor andere RL-strategieen.
3. Voer ablations uit: hybrid zonder MCTS, hybrid met MCTS, en eventueel hybrid met verbeterde transition-supervision.
4. Rapporteer minimaal deze metrics: fill mean/std, placement rate, ms per box, en stabiliteits/proxy-metrics.

## 11. Minimale quality gate voor 24h+ HPC run
1. quick profile end-to-end is groen;
2. checkpoint-resume getest;
3. strict vergelijking faalt niet op ontbrekende strategieen;
4. visualisatie-artifacts worden zonder handmatig ingrijpen gegenereerd;
5. seed/config/version traceability is compleet.

## 12. Conclusie voor expert
Mijn huidige standpunt:
1. `rl_mcts_hybrid` is inhoudelijk de beste eerste thesis-kandidaat;
2. de pipeline is nu veel robuuster voor lange HPC runs;
3. maar world-model transition-supervision is nog een kernrisico voor sterke MCTS-claims.

Ik wil expliciet expertfeedback op:
1. welke physicscomponenten must-have zijn;
2. welke exploratie->exploitatie schedule het meest verantwoord is;
3. hoe we model-based claims methodologisch hard maken met huidige beperkingen.
