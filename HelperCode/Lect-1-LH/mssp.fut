-- Parallel maximum segment sum
-- ==
-- input { [1, -2, 3, 4, -1, 5, -6, 1] }
-- output { 11 }

--
-- compiled input @ mssp-data.in
-- output @ mssp-data.out

-------
-- You may create a random dataset containing integers
-- in [-10 ... 10] with the command:
--      $ futhark dataset --i32-bounds=-10:10 -b -g [64000000]i32 > mssp-data.in
-- After that you may create a reference output, e.g., named mssp-data.out,
--  by running `mssp-seq` on `mssp-data.in`, and after that you may delete the 
--  new line 5, so that both datasets are executed when running 
--  `futhark bench --backend=cuda mssp.fut`
-------

let max (x:i32, y:i32) = i32.max x y

let redOp (mssx, misx, mcsx, tsx)
          (mssy, misy, mcsy, tsy)
        : (i32, i32, i32, i32) =
  let mss = max (mssx, max (mssy, mcsx + misy))
  let mis = max (misx, tsx+misy)
  let mcs = max (mcsy, mcsx+tsy)
  let ts  = tsx + tsy
  in  (mss, mis, mcs, ts)

let mapOp (x: i32): (i32,i32,i32,i32) =
  ( max(x,0)
  , max(x,0)
  , max(x,0)
  , x
  )

let main(xs: []i32): i32 =
  let (x, _, _, _) =
    reduce redOp (0,0,0,0) (map mapOp xs)
  in x
