LoadPlugin("ffms2.dll")
LoadPlugin("JoshUpscale.dll")
FFmpegSource2("original.mp4", fpsnum=30, fpsden=1, threads=1, atrack=-1).AssumeFPS(30, 1)
select_total=framecount() / 200
select_total=select_total * 2
SelectRangeEvery(select_total, 200)
ConvertToRGB24()
input_clip=Last
output_clip=input_clip.PointResize(480, 272).JoshUpscale().LanczosResize(input_clip.width, input_clip.height).Subtitle("JoshUpscale", align=9, text_color=$ffffff, size=25)
input_clip=input_clip.Subtitle("Original", align=7, text_color=$ffffff, size=25)

function side_by_side(clip left, clip right, int "cur_frame", int "speed")
{
  perc = abs(float(cur_frame % (speed * 2)) / float(speed) - 1.0)
  crop_offset = int(perc * right.width)
  (crop_offset <= 2) ? \
  left : \
  (crop_offset == right.width) ? \
  right : \
  StackHorizontal(left.Crop(0, 0, -crop_offset - 2, 0), left.BlankClip(width=2), right.Crop(right.width - crop_offset, 0, 0, 0))
  return Last
}

Animate(0, framecount(), "side_by_side", input_clip, output_clip, 0, 200, \
                                         input_clip, output_clip, framecount(), 200)
