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
--  $ futhark dataset --i32-bounds=-10:10 -b -g [64000000]i32 > mssp-data.in
-- After that you may create a reference output, e.g., named mssp-data.out
--  by running this program, and after that you may delete the new line 5,
--  so that both datasets are executed with `futhark bench --backend=c mssp-seq.fut`
-------

let max (x:i32, y:i32) = i32.max x y

let mssp (xs: []i32) : i32 =
    let best_sum = 0
    let curr_sum = 0
    let (res, _) =
        loop(best_sum, curr_sum) for x in xs do
            let curr_sum = max(0, curr_sum + x)
            let best_sum = max(best_sum, curr_sum)
            in (best_sum, curr_sum)
    in res

let main(xs: []i32): i32 = mssp xs

