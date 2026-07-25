# AO European Elo v2 Dynamic Pilot

`matches.csv`, yalnızca uçtan uca motor davranışını doğrulamak için hazırlanmış
sentetik bir 2026/27 olay akışıdır. Gerçek maç veya tahmin iddiası taşımaz.

Skorlar 90 dakika veya oynandıysa 120 dakika sonundaki saha skorudur; penaltı
atışları skora eklenmez. `pilot-005` bu sözleşmenin penaltı örneğidir. Başlangıç
rating CSV'si `scripts/run_v2_pilots.py` tarafından gerçek 10 takımlı pilotun
v2 statik çıktısından üretilir.

Aktif modelde gol farkı ve Achievement Reserve kapalıdır. Bu nedenle tüm
`goal_multiplier` değerleri `1`, tüm reserve değişimleri `0` olmalıdır.
