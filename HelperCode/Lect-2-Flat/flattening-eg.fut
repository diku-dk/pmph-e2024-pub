----------------------------------------------------------------------------------
-- This is the simple flattening example discussed in slides (lecture notes):
--
--      map (\ i -> map (+(i+1)) (iota i) ) arr 
--
-- or in normalized, nested-parallel form:
--
--     map (\ i -> let ip1 = i+1 in
--                 let iot = (iota i) in
--                 let ip1r= (replicate i ip1) in
--                 in  map (+) ip1r iot
--         ) arr
----------------------------------------------------------------------------------
-- ==
-- entry: main
-- input  { [1i32, 2i32, 3i32, 4i32] }
-- output { [2i32, 3i32,4i32, 4i32, 5i32, 6i32, 5i32, 6i32, 7i32, 8i32] }


import "mk-flag-array"

-- Segmented Scan Helper
let sgmScan [n] 't (op: t -> t -> t) (ne: t)
                   (flags: [n]i32) (arr: [n]t) : [n]t =
  let (_, res) =
    scan (\(x_flag,x) (y_flag,y) -> -- extended binop is denoted $\odot$
             let fl = x_flag | y_flag
             let vl = if y_flag != 0 then y else op x y
             in  (fl, vl)
         ) (0, ne) (zip flags arr)
    |> unzip
  in  res

let main [n] (arr: [n]i32) =               -- arr   = [1, 2, 3, 4]
  -- 1. ip1s = F ( map (\x -> x+1) arr )
  let ip1s = map (\x -> x + 1) arr         -- ip1s  = [2, 3, 4, 5]
  
  -- 2. iots = F ( map  (\i -> (iota i)) arr ) 
  let (flag, flag') =                      -- flag  = [1,2,0,3,0,0,4,0,0,0]
        zip arr ip1s                       -- flag' = [2,3,0,4,0,0,5,0,0,0]
     |> mkFlagArray (map i64.i32 arr) (0,0)
     |> unzip

  let tmp1 = map  (\ f -> if f != 0        -- [0,0,1,0,1,1,0,1,1,1]
                          then 0i32
                          else 1
                  ) flag  
  let iots = sgmScan (+) 0 flag tmp1       -- [0,0,1,0,1,2,0,1,2,3]

  -- 3. ip1rs = F ( map2 (\(i,ip1) -> (replicate i ip1)) arr ip1s )
  let ip1rs = sgmScan (+) 0 flag flag'     -- [2,3,3,4,4,4,5,5,5,5]

  -- 4. F ( map2 (\ip1r iot -> map (+) ip1r iot) ip1rs iots
  in  map2 (+) ip1rs iots                  -- [2,3,4,4,5,6,5,6,7,8]

