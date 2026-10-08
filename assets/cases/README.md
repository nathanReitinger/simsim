# Case images

Each copyright case in [`js/cases.js`](../../js/cases.js) can have a pair of images: **A** is the plaintiff's
work and **B** the defendant's. A case's "Compare the works" button appears once its pair is listed in
`manifest.json`:

```json
{
  "rogers-v-koons": {
    "a": "rogers-v-koons/a.jpg",
    "b": "rogers-v-koons/b.jpg",
    "aLabel": "Art Rogers, Puppies (1985)",
    "bLabel": "Jeff Koons, String of Puppies (1988)"
  }
}
```

- Put the files in a folder named after the case id (`rogers-v-koons/a.jpg`, `rogers-v-koons/b.jpg`).
- JPEG, PNG or WebP all work. About 1,600 px on the long side is plenty and keeps the repository small.
- `aLabel` / `bLabel` are optional captions shown with the results.
- Keys starting with `_` are ignored.
- The holding shown with each case comes from its group in `js/cases.js` (infringement, fair use, no
  infringement). For a mixed or unfinished result, give the case an `outcome` (short label), a `holding`
  (full sentence) and `final: false`, as Cariou v. Prince and Sedlik v. Kat Von D do.

Everything in this repository is published on the public website.
