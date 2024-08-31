------------------------------------------------------------
-- I. the following is a "script" used by automatic testing
------------------------------------------------------------
entry mk_input (n:i64) : [20*n+10]i32 =
   let pattern = [-100i32, 10, 3, -1, 4, -1, 5, 1, 1, -100]
   let rep_pattern = replicate n pattern |> flatten
   let max_segment = iota 10 |> map i32.i64
   in  (rep_pattern ++ max_segment ++ rep_pattern) :> [20*n+10]i32

-- ^ Futhark is implicitly size typed using a syntactic matching
--   method that cannot determine that the result has type 
--   `[20*n+10]i32` as declared by the function signature.
--    
--   The code `:> [20*n+10]i32` casts the result to the specified
--     size type at the cost of a runtime check.
--
--   When applying flattening, you will almost certainly need to
--     creatively use such dynamic casts (`:>`). For example, 
--     `map2` or `zip` expects two array parameters of the same size,
--      hence if the compiler cannot determine that you will need
--      to cast one array to the size of the other.

-------------------------------------------------------
--- II. Parallel Maximum Segment Sum Implementation
-------------------------------------------------------

def max (x:i32, y:i32) = i32.max x y

def redOp (mssx, misx, mcsx, tsx)
          (mssy, misy, mcsy, tsy)
        : (i32, i32, i32, i32) =
  let mss = max (mssx, max (mssy, mcsx + misy))
  let mis = max (misx, tsx+misy)
  let mcs = max (mcsy, mcsx+tsy)
  let ts  = tsx + tsy
  in  (mss, mis, mcs, ts)

def mapOp (x: i32): (i32,i32,i32,i32) =
  ( max(x,0)
  , max(x,0)
  , max(x,0)
  , x
  )

def msspCore (xs: []i32): i32 =
  let (x, _, _, _) =
    reduce redOp (0,0,0,0) (map mapOp xs)
  in x

-------------------------------------------------------------------
-- 3.
-- Benchmarking & validating all entrypoints can be achieved with:
--          $ futhark bench --backend=cuda mssp.fut
--
-- Benchmarking only entrypoint mssp can be achieved with:
--          $ futhark bench --backend=cuda mssp.fut -e mssp
--
-- If validation does not succeed, no runtime is shown and 
--   the `.expected` and  `.actual` files document the differences
-------------------------------------------------------------------


-- Parallel maximum segment sum
-- ==
-- entry: mssp
-- input { [1, -2, 3, 4, -1, 5, -6, 1] }
-- output { 11 }
--
-- compiled random input { [100000000]i32 }
--
-- "Pattern-200000010-elements" script input { mk_input 10000000i64 }
-- output { 45i32 }

--
-- compiled input @ mssp-data.in
-- output @ mssp-data.out

entry mssp(xs: []i32): i32 =
  msspCore xs

----------------------------------------------------------------------------------------
-- The comment above the `mssp` entrypoint demonstrates several ways to specify
--   input datasets and reference results for the purpose of benchmarking and
--   validation directly in Futhark programs (i.e., those are a special type
--   of comment):
--
-- The 1st dataset refers to a small input and reference result, which
--         are both inlined, i.e., the program result is validated against
--         the reference result.
--
-- The 2nd dataset is a randomly generated array of 100 million elements; 
--         since a reference dataset is not provided, no validation is performed
--         Adding "auto output" at the end of the line would mean that program
--         result will be validated against the result obtain by the sequential-C
--         backend---this is useful (only) for finding compiler bugs.
--
-- The 3rd dataset uses a script (i.e., entrypoint `mk_input`) to produce a dataset
--         for which the result is known. 
--
-- The 4th way to benchmark and validate is to use file names to specify the input
--         and output. For example you may create a random dataset containing 
--         integers in [-10 ... 10] by running in the console the command:
--            $ futhark dataset --i32-bounds=-10:10 -b -g [64000000]i32 > mssp-data.in
--         After that, you may create a reference output, e.g., named `mssp-data.out`,
--         by compiling (`$ futhark c mssp-seq.fut`) and running `mssp-seq` on 
--         `mssp-data.in`, i.e., `$ ./mssp-seq < mssp-data.in > mssp-data.out`.
--         This method validates your implementation against a "trusted",
--         algorithmically different implementation.
-- 
-- Note however that the datasets specified for an entrypoint need to span CONTIGUOUS
--   COMMENTED LINES. To make the 4th dataset part of benchmarking/testing you need
--   to remove the blank (empty) line preceding it (once you have created the files).
--
----------------------------------------------------------------------------------------
