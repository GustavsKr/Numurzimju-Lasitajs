# 🚘 Latvijas Automašīnu Numurzīmju Atpazīšana un OCR

Atpazīst un nolas Latvijas automašīnu numurzīmes no attēliem, izmantojot **YOLOv8** objektu detekcijai un **EasyOCR** teksta atpazīšanai Python vidē.

---

![Oriģinālais auto attēls](car.jpg)  
*Oriģinālais automašīnas attēls (piemērs)*

![Izgriezts numurzīmes attēls](cropped_plate.jpg)  
*Izgriezta un apstrādāta numurzīme*

---

✅ **Atpazītā numurzīme:** `FF5418`

---

## 🔍 Ko Šis Skripts Dara

Šis skripts:

- Lejupielādē vienu vai vairākus automašīnas attēlus (no ".jpg" URL)
- Detektē numurzīmi, izmantojot YOLOv8
- Izgriež un apstrādā numurzīmi (pārveido pelēktoņu attēlā, uzlabo kontrastu, samazina troksni)
- Izmanto EasyOCR, lai nolasītu tekstu
- Formatē rezultātu pēc Latvijas numurzīmju standarta (`2 burti + 1–4 cipari`)
- Atgriež biežāko numurzīmi no visiem attēliem

---

## 🔧 Modeļi un Rīki

- **Python**
- **YOLOv8**: numurzīmju detekcijai
- **EasyOCR**: optiskai teksta atpazīšanai (OCR)
- **OpenCV**: attēlu apstrādei
