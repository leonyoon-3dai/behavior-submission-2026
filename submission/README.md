# 제출 정책 및 재현 코드

현재 평가: **002 → 001 → 005, public 0–3 각 1회, 총 12회**. 이후 추가 연구 실험은 하지 않습니다. [최종 제출 순서](../docs/operations/AFTER-SERVER-SUBMISSION.md) · [현재 context](../docs/operations/CURRENT-CONTEXT.md).

최종 manifest는 사용자 확정 A0+RT5task이며, 현재 clean4GPU의 사전 검증을 통과해 총12회를 진행 중이다. actor checkpoint1/2와 task별 auxiliary/normalization은 한 단위로 전환한다. 공식 입력 batch1만 지원하며 출력은 `(1,23)`이다. 정책은RGB/proprio/task_id만 사용하며 privileged 상태는 버린다. 추가depth는 관측에 있어도 사용하지 않는다.

- `contract.py`: 공식 leading batch를 native wrapper용으로 변환. 여러 환경·클라이언트 상태 공유를 허용하지 않는다.
- `policy.py`: 원본A0 또는 기존 설정의RT. 원본26/4/20,RT15/4/15 실행주기 보존. `.arm()`은RT에만 적용한다.
- `server.py`: `/healthz`, `/metadata`,msgpack WebSocket,reset 무응답. 모델은 첫 관측에서lazy load하므로healthz만으로 모델준비를 주장하지 않는다. CPU mock 옵션은 계약검사용이며 제출runner가 거부한다.
- `auxiliary.py`: optimizer를 제거한 inference params. 원본 학습과 optimizer는별도 연구archive에 보존한다.
- `Dockerfile`: 현 정책환경snapshot,실제base digest고정. portablePython+installed packages로 lock한다. 호스트local editable 경로는컨테이너PYTHONPATH로 대체한다.

최종 제출 지원범위는000·001·002·003·005. 007은 보존한 파일럿 모델에만 존재하며 최종 manifest에서 제외한다. 범위 밖은zero action 반환이며지원/성능을 주장하지 않는다. 주최측 partial Docker 처리 요구를 확인할 때 이 동작을 명시한다. 이번 runner는다른task를 실행하지 않는다.

모델은 `/models`,로그는 `/logs` mount. runtime 이미지와model bundle은별도HF archive로복원한다. registry 공개/인증이 아직 확정되지 않으면Docker 접수준비 완료로 표시하지 않는다. 모델 포함 self-contained release의 CPU contract는 확인했지만 registry 게시와 실제24GB 검증은 미완료다. 실행 정책의 weights/seed는 바꾸지 않는다.
