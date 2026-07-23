# Vietnamese TN value-gate report

## Dataset

- Source: `/tmp/omnivoice_vi_tn_pilot.jsonl`
- Records: 200
- SHA-256: `f845a96394f28d3bfd9ca33070af3cf6524dfbe30fc3ad06497e6356b4ad7007`
- Designation: **pilot development set**; no unbiased test accuracy is claimed.
- Historical 115/160: Historical 115/160 result could not be independently reproduced.

## Text baselines

| System | Attempted | Canonical preferred | Acceptable | Status |
|---|---:|---:|---:|---|
| current_rule | 200/200 | 59/200 = 29.5% | 66/200 = 33.0% | 0 runtime errors |
| num2words_vi | 199/200 | 18/200 = 9.0% | 24/200 = 12.0% | 0 runtime errors |
| vietnormalizer | 200/200 | 0/200 = 0.0% | 0/200 = 0.0% | 0 runtime errors |
| soe_vinorm | 200/200 | 93/200 = 46.5% | 105/200 = 52.5% | 0 runtime errors |

Best existing automatic baseline by predeclared canonical preferred accuracy: **soe_vinorm**.
Current rule, VietNormalizer, and soe-vinorm tie for highest coverage at 200/200; soe-vinorm has the best preferred and acceptable accuracy.

Strict matching preserves original casing and spacing; canonical formatting only normalizes NFC, whitespace, and spaces before punctuation. It never lowercases or rewrites words.
VietNormalizer lowercases complete sentences, so it scores zero exact matches under this predeclared case-sensitive protocol; its lower CER shows that this is not equivalent to total verbalization failure.

## Ambiguity and errors

The source schema has no audited `semantic_group`, `reading_style`, ambiguity cluster, or domain. Accordingly, official ambiguous-subset and domain accuracy are unavailable (`cluster_unknown`).
Surface-form slices are diagnostic only and are not called audited ambiguity labels.

Per-role results are stored in `baseline_by_role.csv` and `baseline_by_role.json`. Accuracy for the ambiguous subset is unavailable because the dataset has no trustworthy ambiguity labels.

The largest remaining gaps are contextual identifiers/rooms/phones, plus deterministic coverage for units, ratios, dates, decimals, versions, and punctuation. Package disagreement demonstrates that both parsing/post-processing and context matter. Text-level evidence therefore shows meaningful remaining headroom for a custom system, but the missing ambiguity labels prevent claiming that the headroom is specifically or predominantly contextual.

### 25 informative disagreements

The full machine-readable selection is in `informative_errors.csv`.

| ID | Role | Slice | Raw | Current | Best existing | Gold |
|---|---|---|---|---|---|---|
| phone_008 | EMERGENCY_PHONE | current correct; best existing wrong | Khi cần cấp cứu, hãy gọi 115. | Khi cần cấp cứu, hãy gọi một một năm. | Khi cần cấp cứu , hãy gọi một trăm mười lăm . | Khi cần cấp cứu, hãy gọi một một năm. |
| phone_009 | EMERGENCY_PHONE | current correct; best existing wrong | Khi cần báo công an, hãy gọi 113. | Khi cần báo công an, hãy gọi một một ba. | Khi cần báo công an , hãy gọi một trăm mười ba . | Khi cần báo công an, hãy gọi một một ba. |
| version_ordinal_sport_003 | ERROR_CODE | current correct; best existing wrong | Trang web trả về lỗi 404. | Trang web trả về lỗi bốn trăm linh bốn. | Trang web trả về lỗi bốn trăm linh tư . | Trang web trả về lỗi bốn không bốn. |
| fraction_range_001 | FRACTION | current correct; best existing wrong | Tôi đã ăn 1/2 chiếc bánh. | Tôi đã ăn một phần hai chiếc bánh. | Tôi đã ăn một trên hai chiếc bánh . | Tôi đã ăn một nửa chiếc bánh. |
| fraction_range_003 | FRACTION | current correct; best existing wrong | Có 2/3 số người đồng ý. | Có hai phần ba số người đồng ý. | Có hai trên ba số người đồng ý . | Có hai phần ba số người đồng ý. |
| fraction_range_004 | FRACTION | current correct; best existing wrong | Bài toán yêu cầu tính 5/8. | Bài toán yêu cầu tính năm phần tám. | Bài toán yêu cầu tính năm tháng tám . | Bài toán yêu cầu tính năm phần tám. |
| identifier_001 | IDENTIFIER | current correct; best existing wrong | Mã xác nhận là 105. | Mã xác nhận là một không năm. | Mã xác nhận là một trăm linh năm . | Mã xác nhận là một không năm. |
| location_route_008 | ADDRESS | best existing correct; current wrong | Địa chỉ là 27/5 phố Huế. | Địa chỉ là hai mươi bảy phần năm phố Huế. | Địa chỉ là hai mươi bảy trên năm phố Huế . | Địa chỉ là hai mươi bảy trên năm phố Huế. |
| time_duration_013 | AVAILABILITY | best existing correct; current wrong | Dịch vụ hỗ trợ hoạt động 24/7. | Dịch vụ hỗ trợ hoạt động hai mươi tư phần bảy. | Dịch vụ hỗ trợ hoạt động hai mươi tư trên bảy . | Dịch vụ hỗ trợ hoạt động hai mươi tư trên bảy. |
| year_date_012 | DATE | best existing correct; current wrong | Chúng tôi nghỉ vào ngày 1/5. | Chúng tôi nghỉ vào ngày một phần năm. | Chúng tôi nghỉ vào ngày một tháng năm . | Chúng tôi nghỉ vào ngày mùng một tháng năm. |
| decimal_math_001 | DECIMAL | best existing correct; current wrong | Xác suất bằng 0,5. | Xác suất bằng 0,5. | Xác suất bằng không phẩy năm . | Xác suất bằng không phẩy năm. |
| decimal_math_002 | DECIMAL | best existing correct; current wrong | Kết quả đo là 1,05. | Kết quả đo là 1,05. | Kết quả đo là một phẩy không năm . | Kết quả đo là một phẩy không năm. |
| decimal_math_003 | DECIMAL | best existing correct; current wrong | Giá trị trung bình là 12,04. | Giá trị trung bình là 12,04. | Giá trị trung bình là mười hai phẩy không bốn . | Giá trị trung bình là mười hai phẩy không bốn. |
| decimal_math_006 | DECIMAL | best existing correct; current wrong | Tổng doanh thu là 1.234,56 triệu đồng. | Tổng doanh thu là 1.234,năm mươi sáu triệu đồng. | Tổng doanh thu là một nghìn hai trăm ba mươi tư phẩy năm sáu triệu đồng . | Tổng doanh thu là một nghìn hai trăm ba mươi tư phẩy năm sáu triệu đồng. |
| location_route_009 | ADDRESS | all automatic systems wrong | Cửa hàng ở số 12A đường Láng. | Cửa hàng ở số 12A đường Láng. | Cửa hàng ở số mười hai A đường Láng . | Cửa hàng ở số mười hai a đường Láng. |
| location_route_005 | APARTMENT | all automatic systems wrong | Tôi sống ở căn hộ 1205. | Tôi sống ở căn hộ một nghìn hai trăm linh năm. | Tôi sống ở căn hộ một nghìn hai trăm linh năm . | Tôi sống ở căn hộ mười hai không năm. |
| location_route_006 | APARTMENT | all automatic systems wrong | Căn P2-1508 đang được sửa chữa. | Căn P2-1508 đang được sửa chữa. | Căn Pê hai - một nghìn năm trăm linh tám đang được sửa chữa . | Căn pê hai, mười lăm không tám đang được sửa chữa. |
| identifier_020 | BATCH_ID | all automatic systems wrong | Lô sản xuất mang số 240701. | Lô sản xuất mang số hai trăm bốn mươi nghìn bảy trăm linh một. | Lô sản xuất mang số hai trăm bốn mươi nghìn bảy trăm linh một . | Lô sản xuất mang số hai bốn không bảy không một. |
| identifier_007 | CARD_SUFFIX | all automatic systems wrong | Thẻ của bạn có bốn số cuối là 4821. | Thẻ của bạn có bốn số cuối là bốn nghìn tám trăm hai mươi mốt. | Thẻ của bạn có bốn số cuối là bốn nghìn tám trăm hai mươi mốt . | Thẻ của bạn có bốn số cuối là bốn tám hai một. |
| identifier_016 | CLASS_ID | all automatic systems wrong | Tôi học lớp 12A1. | Tôi học lớp 12A1. | Tôi học lớp mười hai A một . | Tôi học lớp mười hai a một. |
| year_date_010 | DATE | all automatic systems wrong | Hóa đơn được lập ngày 9-1-2021. | Hóa đơn được lập ngày chín tháng một năm hai nghìn không trăm hai mươi mốt. | Hóa đơn được lập ngày chín tháng một năm hai nghìn không trăm hai mươi mốt . | Hóa đơn được lập ngày mùng chín tháng một năm hai nghìn không trăm hai mươi mốt. |
| identifier_006 | ACCOUNT_NUMBER | systems disagree | Số tài khoản là 0011002345678. | Số tài khoản là không không một một không không hai ba bốn năm sáu bảy tám. | Số tài khoản là không không một một không không hai ba bốn năm sáu bảy tám . | Số tài khoản là không không một một không không hai ba bốn năm sáu bảy tám. |
| location_route_010 | ALLEY_NUMBER | systems disagree | Rẽ vào ngõ 8 ở bên phải. | Rẽ vào ngõ tám ở bên phải. | Rẽ vào ngõ tám ở bên phải . | Rẽ vào ngõ tám ở bên phải. |
| location_route_011 | BUS_ROUTE | systems disagree | Tôi đi xe buýt số 32. | Tôi đi xe buýt số ba mươi hai. | Tôi đi xe buýt số ba mươi hai . | Tôi đi xe buýt số ba mươi hai. |
| location_route_012 | BUS_ROUTE | systems disagree | Xe buýt số 105 sắp tới. | Xe buýt số một trăm linh năm sắp tới. | Xe buýt số một trăm linh năm sắp tới . | Xe buýt số một trăm linh năm sắp tới. |

## Paired comparisons

Paired McNemar and bootstrap results are stored in `paired_comparisons.json`. Because rules were edited after inspecting pilot data, p-values are descriptive and not confirmatory.

## Audio value gate

Status: **audio gate pending**.
No product conclusion about raw versus normalized OmniVoice audio is made without listening ratings.

## Current decision

**Decision D — Evidence insufficient.**

Text-level results can justify a 24-case listening pilot, but cannot answer whether normalization improves OmniVoice correctness or naturalness. The smallest valuable next step is to run the frozen 72-sample RAW/BEST_EXISTING/GOLD Colab manifest with one exact reference transcript, then collect blinded human ratings.
