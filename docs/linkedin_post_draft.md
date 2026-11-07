# LinkedIn post draft

Every number in this draft is copied from `results/intel-i5-13400f/headlines.json`, which is computed from the raw
benchmark JSON. Suggested attachments: `docs/carousel.pdf` (7 slides, 1080x1350) or
`docs/figures/hero_light_1200x675.png`.

---

## English

I built a vector search engine in C++20 (HNSW graph, 8-bit quantization, AVX2 kernels, a crash-safe write-ahead log).
Then I benchmarked it against hnswlib and published every result, including the ones where it loses.

What I measured on one desktop CPU (i5-13400F, single thread, same compiler flags and the same tuning grid for both
engines):

- SIFT1M at 95% recall: 11,337 queries/s vs hnswlib's 10,619 (1.07x). At 99%: 5,607 vs 4,916.
- It computes 1,109 distances per query instead of 1,000,000 for a full scan: 902x fewer.
- A graph-construction bug I fixed earlier: across 10 builds, worst-case recall went from 0.641 to 0.998.
- 500 out of 500 random SIGKILL crashes recovered every acknowledged write, and 125 of 125 torn log tails were
  truncated.

Where it loses:

- hnswlib is faster at 90% recall on SIFT1M (1.08x), and at 90% and 99% on GloVe-100 (1.05x).
- Index builds are 1.10x slower on SIFT1M.
- Durable writes are capped at about 212 per second by one fsync per insert, since there is no group commit yet.

Every chart is generated from raw JSON that is committed to the repo, and there is a one-command script to rerun the
whole suite on your machine. If you run it, I'd like to add your machine to the charts.

Code, data and full report: https://github.com/orelshushan556-stack/Vector_Database_Engine (branch `benchmarks`)

#cpp #vectorsearch #databases #performance #benchmarking

---

## עברית

בניתי מנוע חיפוש וקטורי ב-C++20: גרף HNSW, קוונטיזציה ל-8 ביט, קרנלים ב-AVX2 ו-write-ahead log שעמיד בקריסות.
אחר כך מדדתי אותו מול hnswlib ופרסמתי את כל התוצאות, כולל אלה שבהן הוא מפסיד.

מה נמדד על מעבד שולחני אחד (i5-13400F, ת'רד יחיד, אותם דגלי קומפילציה ואותו גריד כיוונון לשני המנועים):

- SIFT1M ב-recall של 95%: ‏11,337 שאילתות לשנייה, לעומת 10,619 ב-hnswlib (פי 1.07). ב-99%: ‏5,607 לעומת 4,916.
- ‏1,109 חישובי מרחק לשאילתה במקום 1,000,000 בסריקה מלאה: פי 902 פחות.
- באג בבניית הגרף שתיקנתי: על פני 10 בניות, ה-recall הגרוע ביותר עלה מ-0.641 ל-0.998.
- ‏500 מתוך 500 קריסות SIGKILL אקראיות שוחזרו בלי לאבד אף כתיבה שאושרה, ו-125 מתוך 125 זנבות לוג קטועים נחתכו.

איפה הוא מפסיד:

- hnswlib מהיר יותר ב-recall של 90% על SIFT1M (פי 1.08), וב-90% וב-99% על GloVe-100 (פי 1.05).
- בניית האינדקס איטית פי 1.10 על SIFT1M.
- כתיבות עמידות מוגבלות לכ-212 בשנייה, כי יש fsync אחד לכל הכנסה ועדיין אין group commit.

כל גרף נוצר מ-JSON גולמי שנמצא ב-repo, ויש סקריפט שמריץ את כל החבילה מחדש בפקודה אחת. אם תריצו אותו אצלכם, אשמח להוסיף
את המכונה שלכם לגרפים.

קוד, נתונים ודוח מלא: https://github.com/orelshushan556-stack/Vector_Database_Engine (ענף `benchmarks`)
