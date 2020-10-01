  ............................................................................
.::                Feladatmegoldások az OE-NIK NIXAP1SBNE tárgyhoz           ::.
--------------------------------------------------------------------------------

A NIXAP1SBNE 'Adatpárhuzamos programozás' tárgy 2020/21/1 félévbeli feladataihoz
találsz itt megoldásokat. Nem biztos hogy hibamentesek, az se biztos hogy elég
lesz benne a magyarázat.

--------------------------------------------------------------------------------

Nincs Visual Studio solution, csak Makefile. A megoldások általában egy
forrásfájlból, amik viszont néhol segédfejléceket is használnak. Ha ezeket VS-be
akarod importálni, akkor húzd át az összes *.h és *.cuh kiterjesztésű fájlt is
a projektmappába.
Ha a Makefile-t akarod használni, akkor lehet (disztrótól és setuptól függ),
hogy átkell írnod az NVFLAGS-t benne és ki kell törölnöd a "-ccbin cuda-g++"
részt.

A forrásfájlok Linux-on biztosan lefordulnak, Windows-on nem garantált.

--------------------------------------------------------------------------------

Néhány megoldás lehet túl advanced, ahhoz képest, hogy hanyadik héthez
tartozik.
Hát... ez van. git gud

--------------------------------------------------------------------------------

A fuzz.sh shell script bizonyos megoldások (pl. 2.cu) tesztelésére való. Ha
tesztelni akarod valamelyik kódot, akkor hívd meg így:
$ ./fuzz.sh EXE_ID
ahol EXE_ID egy exe fájl nevének az eleje. Például a "2.exe" programhoz "2"-es
ID tartozik. 

A profile.sh-nak egy exe teljes nevét átadva az `nvprof` segítségével méréseket
végez a kódon. Első körben az elért occupancy-t méri meg, aztán pedig a
kernelekben és CUDA könyvtárban eltöltött időt.

Értelemszerűen ezek a scriptek csak Linux alatt futnak.

--------------------------------------------------------------------------------

Feladatleírások:

+==============================================================================+
| # Hét    | Forrásfájl       | Feladatleírás                                  |
+==========+==================+================================================+
| 1.1      | 1_1.cu           | Integer vektor szorzása számmal; 1 CPU szálon, |
|          |                  | valamint GPU-n.                                |
+----------+------------------+------------------------------------------------+
| 1.2      | 1_2.cu           | Két integer vektor összeadása                  |
+----------+------------------+------------------------------------------------+
| 1.3      | 1_3.cu           | Component-wise minimumkiválasztás két integer  |
|          |                  | vektorban, eredmény eltárolása egy harmadikba  |
+----------+------------------+------------------------------------------------+
| 2        | 2.cu             | String-ben string keresés                      |
|          |                  | 1. Egy szálon, CPU-n                           |
|          |                  | 2. Egy szálon, GPU-n                           |
|          |                  | 3. GPU-n, 1D N-M+1 variáció                    |
|          |                  | 4. GPU-n, 2D M*(N-M+1) variáció                |
+----------+------------------+------------------------------------------------+
| 3        | 2.cu             | Előző heti feladatok átírása úgy, hogy         |
|          |                  | több blokkot használjanak.                     |
+----------+------------------+------------------------------------------------+
| 4.1      | 4_1.cu           | String-ben string keresés, M*(N-M+1) variáció, |
|          |                  | shared memóriával. (csak a szó shared)         |
|          |                  |                                                |
+----------+------------------+------------------------------------------------+
| 4.2      | 4_2.cu           | Rendezés shared memóriával (egy blokk)         |
+----------+------------------+------------------------------------------------+
