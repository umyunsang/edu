# Wave 2: DigitalPublishing Deep Dive

DigitalPublishing contains a static publishing/course workspace. The strongest public artifact for LinkedIn is the mobile wedding invitation prototype, not the parent repo root.

## Mobile Wedding Invitation

The `mobile-wedding-unrolling-invitation` project adapts a Codrops/Yuriy Artyukh image-unroll WebGL reference into a Korean mobile invitation flow.

Safe claims:

- Static, deployable mobile invitation template.
- Three.js/WebGL image-unroll interaction.
- Korean invitation layout with sample couple/date/venue/map/memory-grid/final reveal content.
- Live Cloudflare Pages demo exists at `ourseason.pages.dev`.
- Project folder documents build/deployment/QA/credit boundaries.

Avoid:

- Fully custom original unroll algorithm.
- Real/final wedding content.
- Framework app.
- Runtime QA beyond the checks actually performed.

## Hook

> WebGL 효과 하나를 모바일 청첩장 템플릿으로 바꾸면서, 인터랙션이 장식이 아니라 읽는 순서를 설계하는 방식이라는 것을 배웠습니다.

## Proof Links

- https://ourseason.pages.dev/
- https://github.com/umyunsang/DigitalPublishing/tree/main/mobile-wedding-unrolling-invitation
- https://github.com/umyunsang/DigitalPublishing
