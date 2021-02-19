LoadPlugin("ffms2.dll")
FFVideoSource("highres.mkv", fpsnum=30, fpsden=1, threads=1, colorspace="YUV420P8")
AssumeFPS(30,1)
LoadPlugin("Dup1.dll")
LoadPlugin("ExactDeDup_x86.dll")
Exist("times_hr.txt") ? \
ExactDedup(firstpass=false, times="times_hr.txt", dupinfo="dupinfo_hr.txt") : \
Dup1().ExactDedup(firstpass=true, times="times_hr.txt", dupinfo="dupinfo_hr.txt")
