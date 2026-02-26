# RL MCTS Hybrid - Thesis Notes (NL)

## 1. Waarom deze strategie eerst trainen
Deze map bevat de meest complexe RL-strategie in de repository: een hiërarchische policy (high-level + low-level) met een world model en optionele MCTS-lookahead.  
Voor een thesis is dit de beste start omdat:
- de architectuur rijk genoeg is voor sterke onderzoeksvragen;
- je direct kunt analyseren wanneer planning (`MCTS`) echt helpt;
- de output (fills, plaatsingsratio, snelheid, robuustheid) goed vergelijkbaar is met DQN/PPO/A2C/PCT/HH.

## 2. Architectuur in 1 minuut
De agent werkt in 3 lagen:
1. **Encoder**: zet observatie (hoogtekaarten + boxfeatures + buffer) om naar een globale state-vector.
2. **High-level policy**: kiest actie-type en context (plaats / skip / reconsider + box/bin-keuze).
3. **Low-level policy (Transformer pointer)**: kiest de beste plaatsingskandidaat uit een kandidaatset.

Daarnaast:
- **World model** voorspelt reward/void/latente volgende toestand;
- **MCTS planner** kan bovenop policy-probabilities een lookahead doen.

Belangrijke bestanden:
- `train.py`: traininglus, curriculum, checkpointing, evaluatie
- `strategy.py`: inferencestrategie in single-bin en multi-bin modus
- `network.py`: modeldefinities
- `mcts.py`: planner
- `candidate_generator.py`: kandidaatconstructie
- `void_detector.py`: trapped-void schatting

## 3. Wat er recent hard gemaakt is (HPC-robuustheid)
In deze versie zijn kritieke failure-points aangepakt:
- skip/reconsider acties worden niet meer geforceerd naar plaatsing;
- evaluatie respecteert high-level box/bin-keuze;
- PPO gebruikt nu actuele high-level action embeddings tijdens update;
- MCTS-pad in multibin inferentie is daadwerkelijk aangesloten (met veilige fallback);
- kandidaat-padding overflow in single-bin pad is gefixt;
- env cleanup bij curriculumwissel en evaluatie is toegevoegd;
- `num_envs` mismatch in `rl_mcts_hybrid` wordt expliciet afgehandeld (single-env per proces).

## 4. Trainingflow (praktisch)
### Fases
1. **Imitation warm-start**  
2. **Curriculum RL** (stage 0 -> stage 3)  
3. **Periodieke evaluatie + checkpointing**

### Belangrijke artefacts
- `logs/metrics.csv`
- `logs/tensorboard/`
- `checkpoints/latest.pt`
- `checkpoints/best_model.pt`
- `checkpoints/final_model.pt`

## 5. Aanbevolen HPC-run (MCTS-first)
Vanuit `python/`:

```bash
python strategies/rl_common/hpc/run_rl_pipeline.py \
  --mode full \
  --profile full \
  --strategies rl_mcts_hybrid,rl_dqn,rl_ppo,rl_a2c_masked,rl_pct_transformer,rl_hybrid_hh \
  --gpus auto \
  --max_parallel 0 \
  --continue_on_error
```

Voor snelle smoke-test:

```bash
python strategies/rl_common/hpc/run_rl_pipeline.py \
  --mode full \
  --profile quick \
  --run_dir outputs/rl_training_smoke/<run_id> \
  --gpus auto
```

## 6. Visualisaties voor thesis (aanbevolen set)
Minimaal:
1. `avg_fill` per strategie met foutbalken (`fill_std`)
2. `placement_rate` per strategie
3. `ms_per_box` (efficiëntie)
4. leercurves (`fill` + moving average)
5. scatter: `fill` vs `ms_per_box` (Pareto)

Voor MCTS-hybrid extra:
1. stage-overgangen vs prestatie
2. aandeel skip/reconsider acties over tijd
3. void-fraction trend als auxiliary quality indicator

## 7. Validatiechecklist voor 24h+ jobs
1. `latest.pt` wordt periodiek geüpdatet
2. run-manifest groeit door alle fases (`train` -> `evaluate` -> `visualize`)
3. geen CSV logger crashes bij extra eval-metrics
4. minstens 1 succesvolle eval-run per strategie met echte checkpoint
5. `comparison/summary_table.csv` bevat alle verwachte strategieën

## 8. Waar physics-simulatie waarschijnlijk het meeste oplevert
Als je extra diepgang wil toevoegen:
1. **Stabiliteit/contactmodel**: steunvlak, kantelmoment, frictie i.p.v. alleen geometrische fit
2. **Compliance/doorbuiging**: zachte dozen en druk-opbouw
3. **Dynamica bij plaatsing**: impact/verschuiving bij neerzetten
4. **Robuustheid tegen sensorfouten**: hoogtekaart-noise en pose-onzekerheid

## 9. Vragen voor expert-review (gericht op verbetering)
1. Is de gekozen reward-samenstelling fysisch realistisch genoeg voor warehouse deployment?
2. Waar is model-based planning (MCTS) zinvol, en waar is het overkill?
3. Welke failure modes ontbreken nog in evaluatie (instabiliteit, beschadiging, throughput)?
4. Welke sim-to-real aannames zijn nu te zwak?
5. Welke extra constraints moet de policy hard afdwingen (gewichtslimieten, labelzijde, fragility)?
6. Welke visualisatie overtuigt een operations team het meest (niet alleen academic metrics)?

## 10. Thesis-positionering (compact)
Een sterke lijn is:
- **Hoofdvraag**: wanneer voegt world-model + planning aantoonbare waarde toe t.o.v. pure policy RL?
- **Methodiek**: MCTS-hybrid eerst stabiel trainen, daarna identieke benchmark tegen de andere RL-strategieën.
- **Uitkomst**: niet alleen beste fill, maar ook stabiliteit, snelheid en reproduceerbaarheid op HPC.
