-- Trivial Examples in Map-Reduce (MR) Form
-- ==
-- compiled random input { [33554432]f32 } auto output

let max (x:f32) (y:f32) = f32.max x y

let id  x = x

--------------------
-- all_p          --
--------------------
let all [n] 't (p : t -> bool) (a: [n]t) : bool =
  map p a |> reduce (&&) true

--------------------
-- sum            --
--------------------
let sum (a : []f32) : f32 =
  reduce (+) 0.0 <| map id a

----------------------------------------
-- fold (\acc x -> acc * (x-2)) 1 arr --
----------------------------------------
let fld (a: []f32) : f32 =
  map (\x -> x - 2f32) a |> reduce (*) 1.0f32

let main [n] (a : [n]f32) : (bool, f32, f32, f32) =
  let p = all (> -10) a
  let m = reduce max 0.0f32 a
  let s = map (\x -> x / m) a |> sum
  let f = fld a
  in (p,m,s,f)


------------------------------------
-- Useful Futhark commands:
--
-- futhark opencl simpleLH.fut
-- futhark bench --backend=opencl simpleLH.fut
-- futhark dataset --f32-bounds=-10.0:10.0 -b -g [64000000]f32 > data.in
