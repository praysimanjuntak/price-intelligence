# EDA Report — MrScraper Price Intelligence

Train shape: (306226, 26)  |  Test (3 days) shape: (25900, 26)


## 1. Column population: anchor vs blank-price test rows

|                          |   anchor_nonnull_% |   blank_nonnull_% |   train_nonnull_% |
|:-------------------------|-------------------:|------------------:|------------------:|
| capturedAt               |              100   |               100 |             100   |
| shopId                   |              100   |               100 |             100   |
| itemId                   |              100   |               100 |             100   |
| modelId                  |              100   |               100 |             100   |
| price                    |              100   |                 0 |             100   |
| priceBeforeDiscount      |              100   |                 0 |             100   |
| promotionId              |              100   |                 0 |             100   |
| cat_id                   |              100   |                 0 |             100   |
| stock                    |                2.3 |                 0 |               1.2 |
| normal_stock             |                2.3 |                 0 |               1.2 |
| raw_discount             |              100   |                 0 |             100   |
| show_discount            |              100   |                 0 |             100   |
| brand                    |               55.3 |                 0 |              65.5 |
| is_free_shipping         |              100   |                 0 |             100   |
| is_pre_order             |              100   |                 0 |             100   |
| item_price_min           |              100   |                 0 |             100   |
| item_price_max           |              100   |                 0 |             100   |
| review_rating            |              100   |                 0 |             100   |
| total_rating_count       |              100   |                 0 |             100   |
| cmt_count                |              100   |                 0 |             100   |
| shop_rating              |              100   |                 0 |             100   |
| shop_response_rate       |              100   |                 0 |              99.6 |
| shop_follower_count      |              100   |                 0 |             100   |
| is_official_shop         |              100   |                 0 |             100   |
| is_verified              |              100   |                 0 |             100   |
| is_preferred_plus_seller |              100   |                 0 |             100   |

## 1b. Is price ~ priceBeforeDiscount * (1 - show_discount/100)?

- rows with pbd>0 & show_discount present: 111,609
- median relative error of discount-reconstruction: 0.0092
- share within 1% of true price: 50.9%
- share with show_discount==0 among these: 0.0%
- price ~ priceBeforeDiscount - raw_discount: median rel err 0.8687, within 1%: 0.1%


## 2. Price distribution (train)

|       |            price |
|:------|-----------------:|
| count | 306226           |
| mean  |      5.23415e+07 |
| std   |      9.1372e+07  |
| min   | 100000           |
| 1%    |      1.2e+06     |
| 5%    |      3.5e+06     |
| 25%   |      9.9e+06     |
| 50%   |      2.05e+07    |
| 75%   |      5.9e+07     |
| 95%   |      1.8e+08     |
| 99%   |      3.899e+08   |
| max   |      1.66e+09    |

- zeros: 0  |  negatives: 0
- min: 100,000  max: 1,660,000,000
- log10 range: [5.00, 9.22]


## 3. Entity history depth in train (cold-start risk)

|    | col     |   n_unique_train |   median_hist_rows |   p90_hist_rows |   test_unique |   test_unseen_in_train |   test_unseen_% |
|---:|:--------|-----------------:|-------------------:|----------------:|--------------:|-----------------------:|----------------:|
|  0 | modelId |             6286 |                 33 |             122 |          3871 |                      2 |            0.05 |
|  1 | itemId  |             1594 |                 64 |             425 |           953 |                      0 |            0    |
|  2 | shopId  |              219 |                228 |            2908 |           150 |                      0 |            0    |
|  3 | cat_id  |               26 |               5034 |           34615 |            22 |                     22 |          100    |
|  4 | brand   |              376 |                135 |            1090 |            72 |                      0 |            0    |

## 3b. Coverage of blank test rows by train history

- blank rows with modelId seen in train: 99.99%
- blank rows with itemId seen in train: 100.00%
- blank rows with shopId seen in train: 100.00%


## 4. Per-modelId price volatility in train

- modelIds with >=3 obs: 6,145
- median coefficient of variation: 0.0000
- share of modelIds with CV < 1% (essentially constant price): 85.7%
- share with CV < 5%: 91.2%
- median max/min price ratio: 1.0000


## 5. Anchor representativeness (per test day)


**2025-03-22** — anchors=100, blank=9214
- anchor price median=13,850,000, mean=43,338,000
- anchor distinct shops=37, distinct cats=20

**2025-03-23** — anchors=100, blank=7953
- anchor price median=16,750,000, mean=32,736,000
- anchor distinct shops=41, distinct cats=16

**2025-03-24** — anchors=100, blank=8433
- anchor price median=11,100,000, mean=26,317,000
- anchor distinct shops=34, distinct cats=18
