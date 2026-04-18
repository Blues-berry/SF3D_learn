# Pool B Signal Check

| source | controlled lighting | mask | normals | albedo | specular separation | suitability | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLATverse | yes | yes | yes | yes | not explicit | high | project page exposes full-bright, OLAT, env relit, masks, normals, diffuse albedo |
| OpenIllumination | yes | yes | unknown | unknown | unknown | medium-high | Hugging Face card clearly states 13 patterns + 142 OLAT and object/composition masks |
| ICTPolarReal | yes | unknown | yes | yes | yes | high if accessible | preprint advertises polarized diffuse/specular decomposition and real-object inverse rendering signals |

## Readout

- best low-risk Pool-B pilot source: `OpenIllumination` because the license is already clear (`CC BY 4.0`)
- best high-signal Pool-B source if access is granted: `ICTPolarReal`
- most complete synthetic-style decomposition source from the public project page: `OLATverse`
