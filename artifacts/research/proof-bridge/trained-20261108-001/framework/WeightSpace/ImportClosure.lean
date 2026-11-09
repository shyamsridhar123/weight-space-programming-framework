import Lean.Elab.ParseImportsFast
import Lean.Util.Path

/-! Use Lean's own header parser, including comments, strings, modules and prelude. -/
namespace WeightSpace.ImportAudit

structure Row where
  moduleName : String
  source : String
  imports : Array String
  deriving Lean.ToJson

def main (args : List String) : IO UInt32 := do
  let [configPath] := args | throw <| IO.userError "Supply one import-audit configuration file"
  let .ok config := Lean.Json.parse (← IO.FS.readFile configPath)
    | throw <| IO.userError "Invalid import-audit JSON"
  let .ok roots := config.getObjValAs? (Array String) "roots"
    | throw <| IO.userError "Missing source roots"
  let .ok seeds := config.getObjValAs? (Array String) "seeds"
    | throw <| IO.userError "Missing seed modules"
  let searchPath : Lean.SearchPath := roots.toList.map System.FilePath.mk
  let mut pending := seeds.map String.toName
  let mut seen : Std.HashSet Lean.Name := {}
  let mut rows : Array Row := #[]
  while !pending.isEmpty do
    let name := pending.back!
    pending := pending.pop
    if seen.contains name then continue
    seen := seen.insert name
    let source ← Lean.findLean searchPath name
    let header ← Lean.parseImports' (← IO.FS.readFile source) source.toString
    let imports := header.imports.map (fun entry => entry.module)
    rows := rows.push { moduleName := name.toString, source := source.toString,
                        imports := imports.map Lean.Name.toString }
    pending := pending ++ imports
  IO.println (Lean.toJson rows |>.compress)
  return 0

end WeightSpace.ImportAudit

def main := WeightSpace.ImportAudit.main
