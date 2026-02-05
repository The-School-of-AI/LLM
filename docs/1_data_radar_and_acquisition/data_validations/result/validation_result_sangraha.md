# Parquet Schema Validation Report

**Generated:** 2026-02-05 10:44:42

**S3 Path:** s3://t1-dataacquisition-datasets/datasets_prod/sangraha/hin/

**Total Records Validated:** 21

## Validation Summary

- ✅ **Valid Records:** 21
- ❌ **Invalid Records:** 0
- 📊 **Success Rate:** 100.0%

## Expected Schema

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique record identifier |
| hash | string | SHA-256 hash of text content (for deduplication) |
| dataset | string | Source dataset name |
| domain | string | Content domain (web, literature, education) |
| source | string/null | Source identifier (for Dolma) |
| text | string | Main text content |
| language | string | Full language name |
| metadata | dict | Additional dataset-specific fields |
| added | string/null | ISO timestamp when added |
| created | string/null | ISO timestamp of creation |
| version | string/null | Dataset version |

## Detailed Validation Results

### Record 1 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00536.parquet`

**Record Index:** 6861

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "e55917c8ae9b31ba8fdea48f7963e17ac9056a82",
  "hash": "c441f080724d7d2a743923ac089e5830fadca7401aaec86ef28281bea701f496",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "कानपुर (ब्यूरो) लंगर इंचार्ज मंजीत ङ्क्षसह सागरी की अगुवाई में 400 से ज्यादा सेवादार, दीप सेवादल के प्रधान आत्मबीर ङ्क्षसह ने व्यवस्था संभाली। भव्य पालकी में श्री गुरु ग्रंथ साहिब विराजमान रहे। लुधिया",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"web\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 2 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00536.parquet`

**Record Index:** 2911

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "5c8154cc3558a7d7f482b5123585c1aec24771210cf9a21c880b150574993c68",
  "hash": "e610dfb3a0b5028fc30b3bdded58f23acd0269149876d77f4130933cbd13d282",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "अपने फन का... : 427\nएक दिन मैं हेम बाबू को डेरे पर छोड़कर कुछ काग़ज़ खरीदने बाज़ार गया था। वहाँ देखा कि दुकान के अन्दर तख़्त पर बैठा हुआ एक आदमी ज़ोर-ज़ोर से इस दिन का अख़बार पढ़ रहा था, और कई बेकार आ",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"pdf\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 3 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00536.parquet`

**Record Index:** 9555

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "64c646fb9bd26a63545ec1185a8c1d7547a284787d7338336186471b6c091e22",
  "hash": "1197143585ef3bfb8687694fa9c1aa20836445de74d45471b10b84033801a273",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "● Narrator as spectator of monkeys\n* \"Monkey mothers\"\n* \"Older monkeys\"\nRailway Station\n\"Younger fraternity\"\nCollect food\nEnjoying a good old scratch\nSkilfully sitting on branches...waiting to pounce ",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"speech\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 4 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00353.parquet`

**Record Index:** 3736

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "27be0f39c8f01224431cc0fac0df72af20a1df46",
  "hash": "a57fd837b75efb1a915647285cc81db76b86afeb0123938513ab237f9d0abb0c",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "डा. वाईएस परमार बागवानी एवं वानिकी विश्वविद्यालय नौणी ने 176 लावारिस गायों को खूंटों में बांध लिया है। इन्हें विवि परिसर के अलावा प्रदेश भर के बागवानी अनुसंधान केंद्रों में बाधा है। इस यूनिवर्सिटी ने ",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"web\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 5 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00353.parquet`

**Record Index:** 8719

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "f51d9ef9fd17995a1d3a2fbfd002726d4cae982c",
  "hash": "52473fe6e0f742d7b76c304b0afb49657a62f07084966f4ecaa00ba08549e553",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "रिक्त पदों का नाम -1. सब्जेक्ट स्पेशलिस्ट (Subject Specialist)\n2. असिस्टेंट इंजीनियर (Assistant Engineer)\n3. सब इंजीनियर (Sub Engineer)\n4. असिस्टेंट प्रोग्रामर (Assistant Programmer)\n5. डाटा एंट्री ऑप",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"web\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 6 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00353.parquet`

**Record Index:** 9075

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "e8b5e71bbc196da5741b0183b7aaa96c419b3659",
  "hash": "7694bcf54c0ad6d1f243462873748d2a7c243b8333fe40f4c968457439dea93d",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "मुंबईः राजस्थान के उदयपुर में भाजपा की निलंबित प्रवक्ता नूपुर शर्मा के समर्थन में पोस्ट करने पर कन्हैया लाल की कट्टरपंथियों द्वारा निर्मम हत्या और अमरावती में भी ऐसे ही हत्या के बीच नागपुर में एक परिव",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"web\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 7 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00225.parquet`

**Record Index:** 4068

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "101543824ec8b207d01b7ae5c79bb144a2b26a70",
  "hash": "9fb81c439e890e45bfad2349185a200dfa60a7a561dc01778caf81fb20ce0f05",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "बाबा बेदी पेट्रोल पंप स्थित एसबीआई ब्रांच में अब बुजुर्गो व दिव्यांग लोगों के लिए लिफ्ट सुविधा उपलब्ध होगी। इसके अलावा बैंक को रेनोवेट कर ग्राहकों के बैठने के लिए भी बढिय़ा व्यवस्था की गई है। इससे पहल",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"web\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 8 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00225.parquet`

**Record Index:** 9905

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "e77097b859711623d0c31815993baad457ec0e00",
  "hash": "d40fac230afc8097efba2cd4b62bcbb78e9031e44d8e16addbd2441dbd97ad25",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "बैठक के बाद उप राज्यपाल बैजल ने भी ट्वीट किया, \"दिल्ली के मुख्यमंत्री ने मुझसे राजनिवास में मुलाकात की। मैंने उन्हें भरोसा दिलाया है कि कानून का उल्लंघन करने वालों के खिलाफ कड़ी कार्रवाई की जाएगी। \" क",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"web\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 9 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00225.parquet`

**Record Index:** 9100

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "7a8da1df75f0c5d2fc91a469a3017d5ce77037ca",
  "hash": "a735317f5a9b45e474014026136fb1148440d6d57237eaa205fe00cd0e6f0f51",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "अमरूद विटामिन ए का एक बहुत अच्छा स्रोत है, इसलिए आंखों के लिए भी फायदेमंद है. यह शरीर को स्वस्थ रखने में भी सहायक है और रक्तचाप को भी कम करता है. साथ ही खून की तरलता को बनाए रखता है. यह विटामिन सी की ",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"web\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 10 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_01713.parquet`

**Record Index:** 9671

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "66bed780f73953b14fc52be893b56d463df3fa5d",
  "hash": "9a207a50d4ea80ac09242928b65db7197538c611d4111285ef5453cb3a43a802",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "सुहेलदेव भारतीय समाज पार्टी के प्रमुख ओम प्रकाश राजभर का दावा है कि मंगलवार (10 मई, 2022) को लाठी-डंडा लेकर 10-12 लोग उनको मारने के लिए आए थे। उन्होंने इस घटना को लेकर भारतीय जनता पार्टी पर भी हमला बो",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"web\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 11 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_01713.parquet`

**Record Index:** 3432

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "7c4e6b23c690b32308101e28c0f8b4555f1f6979c76e8cb165a915025e468b50",
  "hash": "ea8b9c11bce58ba84f51eaf2750a5944c7f138a06e95b3e82bdc314c8ac9cc7f",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "THE GAZETTE OF INDIA : EXTRAORDINARY परिशिष्ट 6--जारो\nकी तारीख को या उसके कर सका हैं :खुले सामान्य लाइन के अन्तर्गत आयात पर लागू शर्ते :\n( 1 ) गभी वास्तविक मान की निकासी के समय शुल्क प्राधिकारियों को ",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"pdf\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 12 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_01713.parquet`

**Record Index:** 2344

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "b9609e25959cb706ad9a5f55313893c2f3157488db5b0500dc4e67882032a7cf",
  "hash": "ce31045830bf10cdd97dca39aa55096b9df06c1e906e7068a70d7257e72c982b",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "भावकों के समाराधन से इस काव्यप्रकार में नवीन छंदों, गीतों एवं अभि जय के नवीन प्रयोगों को विकास का अवसर मिला ।\nअभिनेय होने के कारण एक ओर गीतों में सरसता और संगीतमयता • लाने का प्रयास होता रहा और इस उद्",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"pdf\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 13 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00448.parquet`

**Record Index:** 3176

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "c4ff7ebdfe1869e6891472054d5ff2aab295fef1c7fa6f458b702ab4e2102b7b",
  "hash": "886ade376e0e66828b0f74f66e72be26daeefe1418ffd0c52ed7b0dc2709ea60",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "देश की ही हैसियत में लाना चाहता है वह उसे सभ्यता के पैमाने में गिराना चाहता है ।\" और हिंदुस्तान में अंग्रेजों ने ठीक यही चीज़ करने की जी-जान से, बराबर कोशिश की और हिंदुस्तान में सौ पचास बरस की हुकूमत ",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"pdf\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 14 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00448.parquet`

**Record Index:** 1266

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "99dd4a7b4cbcdd156522c1631250a22192fe245991359c18d10815f697c17887",
  "hash": "adb1f9d3f42b45616d8f3d66e5ab5464eb0e8d16d0f82aa7b19c53c0413af523",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "अजमेर सूबे का यह प्रान्तीय शासन संगठन ईसा की १८ वीं शताब्दी के प्रारम्भ में फर्रुखसियर के शासन काल तक बहुत ही थोड़े हेर-फेर के साथ ऐसा ही चलता रहा । इस लम्बे काल में जो दो महत्वपूर्ण परिवर्तन किये गए ",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"pdf\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 15 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00448.parquet`

**Record Index:** 274

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "6ded60cf2e96991815f16ab39bc9205c7bab4f57",
  "hash": "35fa0849577e085c99b52d75aae03877790791ce4f1ecf80c9cff3bc7ae9b3ab",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "नई दिल्लीः कोरोना महामारी और अब रूस-यूक्रेन में युद्ध के चलते खाद्य तेलों के दामों में बढ़ोतरी हुई है. हालांकि सरसों के तेल कि कीमतों पर अभी इसका प्रभाव नहीं पड़ा है, मगर विशेषज्ञ मानते हैं कि आने वाल",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"web\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 16 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00242.parquet`

**Record Index:** 4168

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "5919a620fd2fd520761193168f4dfc8a736d1017",
  "hash": "b4414eb4276571b29defb4bb98e12247f1362b2c90c487ddac473a6b5a5b57ab",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "कई शोधकर्ताओं के अनुसार, yl मछली पानी के नीचे किंगडम का सबसे दिलचस्प प्रतिनिधियों में से एक है। अतिशयोक्ति के बिना, इस सार अद्वितीय कहा जा सकता है। हमारा लेख जीवन शैली, व्यवहार के बारे में बताता है और",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"web\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 17 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00242.parquet`

**Record Index:** 7529

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "43ce9cd171fa8390c7a05c5b2c5991df0d38ffda",
  "hash": "8fb358457a9877ad976a3fab0d8dcfa08e68eda1e7e990c5f3c8757ab74fde90",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "आइसीसी क्रिकेट समिति के प्रमुख अनिल कुंबले ने कहा है कि लोकल अंपायरों के पास ज्यादा अनुभव नहीं होहता और इस वजह से उन्होंने टेस्ट क्रिकेट में अतिरिक्त रिव्यू का सुझाव दिया है।\n नई दिल्ली, प्रेट्र। कोवि",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"web\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 18 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_00242.parquet`

**Record Index:** 5546

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "f2b8b3480d8fa9d56e5cd9e1a39d06ab7ead4f4c",
  "hash": "d0899dbe559a98339928d26e76538ed338d7bc662269f4c6e40243881f1993dd",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "लखनऊः संजय गांधी पीजीआई (SGPGI) के डॉ. ज्ञानचंद ने 8 साल के एक बच्चे का एड्रिनल ग्रंथि का ट्यूमर रोबोटिक्स सर्जरी (छोटे से छेदों) से निकाल दिया. दावा किया गया है कि यह सर्जरी उत्तर प्रदेश और संपूर्ण भ",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"web\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 19 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_01067.parquet`

**Record Index:** 207

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "7722e3d58dffa11e308c2bc1b215bcc4d393e95b",
  "hash": "fec5e11b2d2459e7da662288c0cbcd3337a1dcf62201cc3adfb15dbf57296f02",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "छपरा. रात में बड़े भाई की साली के कमरे में मिले छोटे भाई की गांववालों ने सुबह शादी करा दी। लड़की के कमरे से लड़का पकड़ाए जाने की खबर के बाद लोगों की भीड़ उमड़ पड़ी। रात में ही पंचायत बैठी और लड़के के ",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"web\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 20 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_01067.parquet`

**Record Index:** 3233

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "5b61535e64bf2399cf8a2ab4aab14a4197810d7d4e23d9af12b77e4443b1edaa",
  "hash": "6074ee0e2624ddcaeff8b77c6a030680d99fa2253ee87f999b845379839e082b",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "कार्ल मार्क्स का जन्म ५ मई, १८१८ को वियेर नगर ( प्रशा के राइन प्रान्त ) में हुआ था। उनके पिता एक यहूदी वकील थे जिन्होंने १८२४ में प्रोटेस्टेंट मत अंगीकार किया था । यह परिवार समृद्ध और सुसंस्कृत था, पर",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"pdf\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

### Record 21 - ✅ VALID

**File:** `datasets_prod/sangraha/hin/parquet/records_01067.parquet`

**Record Index:** 8413

#### Field Validation

| Field | Expected Type | Actual Type | Valid | Issues |
|-------|---------------|-------------|-------|--------|
| id | string | str | ✅ | - |
| hash | string | str | ✅ | - |
| dataset | string | str | ✅ | - |
| domain | string | str | ✅ | - |
| source | string/null | NoneType | ✅ | - |
| text | string | str | ✅ | - |
| language | string | str | ✅ | - |
| metadata | dict | str | ✅ | Dict stored as JSON string |
| added | string/null | NoneType | ✅ | - |
| created | string/null | NoneType | ✅ | - |
| version | string/null | NoneType | ✅ | - |

#### Record Data Preview

```json
{
  "id": "a7aa9d566a97e9a6a7d59d55be7270cf2e2c2d37",
  "hash": "e691e6799a0ee08a167f6fe269695eb5ffa903a7ac6b57cfe0751adcb410e056",
  "dataset": "Sangraha",
  "domain": "web",
  "source": null,
  "text": "प्रयागराज के प्रधान डाकघर स्थित पासपोर्ट सेवा केंद्र पर प्रतिदिन 100 लोगों को पासपोर्ट बनवाने के लिए अप्वाइमेंट दिया जा रहा है। हालांकि जानकारी के अभाव व अधूरे कागजातों के कारण महज 50 से 60 लोगों का ह",
  "language": "Hindi",
  "metadata": "{\"language_code\": \"hin\", \"type\": \"web\"}",
  "added": null,
  "created": null,
  "version": null
}
```

---

