# Ilia + RT-EXPO-FT: 실제 추론·학습 구성 감사

2026-10-11에 main/preparation history, 현재 실행중인 policy Docker의 upstream 소스, 최종 manifest와 저장된 추론로그를 확인했다. read-only 감사이며 actor/보조망/평가 실행은 바꾸지 않았다.

## 결론

최종 정책은 **Ilia actor 전체를 동결한 채, Ilia의 stage voting·correlation-aware inpainting·후처리를 유지하고, RT의 후보선택과 작은 residual edit를 추가한 시스템**이다. Ilia actor만 가져온 단순 정책도 아니며 RT-EXPO upstream의 모든 학습 구성요소를 온라인에서 사용하는 경로도 아니다. FT는 보조망 학습이며 최종 Ilia actor 파라미터 업데이트가 아니다.

## Ilia에서 유지한 것과 달라진 것

|구성|최종 상태|근거/의미|
|---|---|---|
|PaliGemma vision/VLM, action expert|유지·동결|create_trained_policy(pi_behavior_b1k_fast), native model.sample_actions 호출|
|task embeddings, task/stage fusion|유지·동결|[task_id,current_stage]가 prompt token으로 입력됨|
|VLM stage predictor|유지·동결|선택한 base 후보의 subtask_logits를 wrapper에 반환|
|KV cross-layer transform|유지·동결|runtime config use_kv_transform=True, native sample_actions에서 적용|
|상관 잡음과 action correlation matrix|유지|native sampler 사용; explicit training noise는 Cholesky 변환|
|delta action/per-timestamp normalization|유지|native→physical→critic frame 변환과 decode/rebase 사용|
|rolling inpainting|유지하되 RT scheduling 적용|남은4개 prefix로 다음chunk를 비동기 계산|
|stage voting|유지|history_len3, votes_to_promote2; wrapper update_current_stage 실행|
|hand-coded correction/eval tricks|활성 유지|apply_eval_tricks 기본True, 선택/편집된 action 이후에도 적용|
|원본26개 action을20스텝으로 압축|RT에서 비활성|원본A0 config26/4/20 vs RT15/4/15; should_compress=False|
|gripper variation에 따른 압축 중지|RT에서 실행효과 없음|압축 자체가 비활성이라 해당분기 들어가지 않음|
|FAST auxiliary token branch|학습전용, 온라인추론 미사용|native sample_actions는 FAST tokens 입력을 금지함. 기존학습 효과는 weights에 남음|
|원본4개 checkpoints task switcher|일부route 선택|최종001=checkpoint2,002/005=checkpoint1; 전체50task/4checkpoint 시스템 복제 아님|

stage voting은 행동후보의 다수결이 아니다. 최근3회 stage예측을 모아 다음stage로 넘어가는 규칙이며, RT 행동후보 선택은 별도로 critic Q argmax를 사용한다.

## RT 추론의 정확한 경로

1. 동결 Ilia native actor가30-step 후보2개(N2)를 생성한다. 실제 실행 action은23D이며 native padded action은32D다.
2. 직전chunk에4개action이 남았을때 prefix4를주고 다음후보를별도thread에서미리계산한다. 경계에서최신관측과anchor로rebase한다.
3. delay4 이후 실행창15스텝을 가져온다. RGB3카메라를 EXPO encoder에 넣고 별도critic용상태와 함께 평가한다.
4. 첫번째base후보에 edit actor가 residual1개를생성한다. edit_scale0.01(정규화공간), mask/edit_action_xyzg=False다. 즉 최종후보는 base2개+edited1개=3개다.
5. target critic2개의Q중최솟값을 후보점수로 쓰고,3후보의argmax를선택한다. 평균/투표형action ensemble은 아니다.
6. selected15+동일base후보carry4를decode하여19×23 action을wrapper로반환한다. 선택edited인경우에도 stage_logits는그편집의base후보것이다.
7. Ilia correction rules와stage voting이그대로적용되고15스텝을실행한다. correction이발동하면 RT선택결과를덮어쓸수있다.

내부learned_Q_values는critic예측이며공식종료Q와다르다.30Hz_prefix_budget_met도공식평균1FPS기준과다르다. 이번실행의slow 약0.15초는4/30=0.133초보다커서30Hz예산미달로그가있지만 실제시뮬레이터는그보다느리므로같은wall-FPS판정이아니다.

## 학습과 온라인 사용은 구분

train_base_actor=False로 actor update를건너뛴다. actor초기/최종SHA일치assert를실행한다. 최종task별보조망1000updates는원본task별시연20episode replay, batch8이며다른task보조망을그대로전이한것이아니다. sampling audit는8000draws와20episode사용을기록한다.

|EXPO 구성|학습/갱신|온라인 직접 사용|
|---|---|---|
|critic|gradient 학습|fast선택은targetcritic사용,online critic직접평가아님|
|target_critic|critic의EMA, tau0.005|사용:후보3개평가|
|batch_encoder|학습(freeze_critic_encoder=False)|사용:RGB특징|
|edit_actor|학습|사용:residual후보|
|filter_critic|학습,학습시noise 후보 filter에사용|**현재sample_pre_cache/fast_select추론경로에서는미사용**|
|temp|학습:edit entropy계수|현재온라인fast_select직접미사용;edit학습에간접영향|
|Ilia actor전체/stage/KV/task heads|최종 RT 학습에서 갱신 없음|이미학습된동결weights사용|

freeze_encoder=True만보고모든encoder동결이라고해석하면안된다. 별도EXPO batch_encoder는freeze_critic_encoder=False로학습한다. filter/temp는inference NPZ에도포함·복원되지만복원과실제호출은다른것이다. final NPZ에는critic10,target10,filter10,encoder34,edit12,temp1 parameter leaves가각task에존재한다.

## correction 규칙의 중요한 제한

runtime correction_rules의task001/002/005는양손always-open task목록에있다. 단,실제발동은current gripper state<-0.98일때이며항상매스텝손을여는뜻이아니다. 발동하면chunk를현재state로채우고해당손gripper1.0으로바꾼다. 이번추론로그는그립상태/규칙발동횟수/실제stage전이를담지않으므로이규칙이001실패의원인이라고확정할수없다. 후보Q만으로최종실행action을완전히설명할수도없다.

## history의 구분

초기실험에는추가학습한F1 actor위에RT를올린연구가있다(훈련기록F1+RT-EXPO-FT). 그경우RT단계는F1 actor를동결했어도F1자체는과거추가학습actor였다. **현재최종manifest는원본checkpoint1/2 A0 actor + task별RT보조망**으로,초기F1실험과구분한다. 이후actor를20demo로추가학습하자는제안은사용자가보류했고실행하지않았다. 따라서역사의모든actor가추가학습된적없다고쓰면부정확하다.

## 근거

- 현재 [policy factory](../../submission/policy.py), [native actor adapter](../../adapters/ilia_expo_actor.py), [RT inference](../../adapters/ilia_expo_inference.py), [learner config](../../adapters/ilia_expo_learner.py), [training script](../../scripts/train_ilia_rt_expo.py), [manifest](../../submission/model-manifest.final.json)
- [원본 Ilia repository](https://github.com/IliaLarchenko/behavior-1k-solution), runtime업스트림소스추출은outputs/clean-server-20261011/inference-audit에보존
- history commits2478abe(동결Ilia RT측정),3cb3dae(원본task자료/독립A0대조)
- preparation의experiments/task0/ilia-rt-expo-comparison-20261008T0932/training-report.json은F1 actor_initial/finalSHA동일,actor_updates0,auxiliary1000기록
- experiments/task0/five-task-20261009/task001-restored-aux-verification.json 및preparation의task001/002/005-training-draw-audit.json
