import type { GameState, Move } from './types';
import {
  applyMove,
  getAllLegalMoves,
  isInCheck,
  pieceType,
  pieceColor,
  getGameResult,
} from './chess';

// ──────────────────────────────────────────────────────────────────────────────
// Piece values
// ──────────────────────────────────────────────────────────────────────────────

const PIECE_VALUE: Record<string, number> = {
  P: 100,
  N: 320,
  B: 330,
  R: 500,
  Q: 900,
  K: 20000,
};

// ──────────────────────────────────────────────────────────────────────────────
// Search helpers
// ──────────────────────────────────────────────────────────────────────────────

const MAX_PLY = 64;
const killerMoves: (Move | undefined)[][] = Array.from({ length: MAX_PLY }, () => [undefined, undefined]);
const historyHeuristic: Record<string, number> = {};

function moveKey(m: Move): string {
  return `${m.from.r}${m.from.c}${m.to.r}${m.to.c}${m.promotion ?? ''}`;
}

function sameSquare(a: { r: number; c: number }, b: { r: number; c: number }): boolean {
  return a.r === b.r && a.c === b.c;
}

function areMovesEqual(a?: Move, b?: Move): boolean {
  if (!a || !b) return false;
  return (
    sameSquare(a.from, b.from) &&
    sameSquare(a.to, b.to) &&
    a.promotion === b.promotion
  );
}

function recordKiller(ply: number, move: Move) {
  if (ply >= MAX_PLY) return;
  const [k1, k2] = killerMoves[ply];
  if (areMovesEqual(k1, move)) return;
  killerMoves[ply][1] = killerMoves[ply][0];
  killerMoves[ply][0] = move;
}

function recordHistory(move: Move, depth: number) {
  const key = moveKey(move);
  historyHeuristic[key] = (historyHeuristic[key] ?? 0) + depth * depth;
}

// ──────────────────────────────────────────────────────────────────────────────
// History / repetition
// ──────────────────────────────────────────────────────────────────────────────

type HistoryState = {
  fenKeys: string[];
  moves: Move[];
  noProgressPly: number;
};

const MAX_HISTORY = 24;

function cloneHistory(h: HistoryState): HistoryState {
  return {
    fenKeys: [...h.fenKeys],
    moves: [...h.moves],
    noProgressPly: h.noProgressPly,
  };
}

function boardKey(state: GameState): string {
  const rows: string[] = [];
  for (let r = 0; r < 8; r++) {
    let row = '';
    for (let c = 0; c < 8; c++) {
      row += state.board[r][c] ?? '.';
    }
    rows.push(row);
  }
  return `${rows.join('/')}_${state.turn}`;
}

function isReverseMove(current: Move, prev?: Move): boolean {
  if (!prev) return false;
  return (
    sameSquare(current.from, prev.to) &&
    sameSquare(current.to, prev.from) &&
    current.piece === prev.piece
  );
}

function isRookShuffle(move: Move, prev?: Move): boolean {
  if (!prev) return false;
  if (!move.piece || pieceType(move.piece) !== 'R') return false;
  if (!prev.piece || pieceType(prev.piece) !== 'R') return false;
  return isReverseMove(move, prev);
}

function updatesNoProgress(move: Move, prevNoProgress: number): number {
  const pawnMove = move.piece ? pieceType(move.piece) === 'P' : false;
  if (pawnMove || move.captured || move.promotion) return 0;
  return prevNoProgress + 1;
}

function pushHistory(history: HistoryState, stateAfterMove: GameState, move: Move): HistoryState {
  const next = cloneHistory(history);
  next.moves.push(move);
  if (next.moves.length > MAX_HISTORY) next.moves.shift();

  next.fenKeys.push(boardKey(stateAfterMove));
  if (next.fenKeys.length > MAX_HISTORY) next.fenKeys.shift();

  next.noProgressPly = updatesNoProgress(move, next.noProgressPly);
  return next;
}

// ──────────────────────────────────────────────────────────────────────────────
// NNUE
// ──────────────────────────────────────────────────────────────────────────────

interface NNUEWeights {
  fc1_weight: Float32Array[];
  fc1_bias: Float32Array;
  fc2_weight: Float32Array;
  fc2_bias: number;
}

class NNUEEvaluator {
  private weights: NNUEWeights | null = null;
  private accumulator: Float32Array = new Float32Array(256);
  private inputSize = 768;

  private pieceMap: Record<string, number> = {
    wP: 0, wN: 1, wB: 2, wR: 3, wQ: 4, wK: 5,
    bP: 6, bN: 7, bB: 8, bR: 9, bQ: 10, bK: 11,
  };

  public async loadWeights(url: string): Promise<void> {
    const resp = await fetch(url);
    const buffer = await resp.arrayBuffer();
    const data = new DataView(buffer);
    let offset = 0;

    const magic =
      String.fromCharCode(data.getUint8(0)) +
      String.fromCharCode(data.getUint8(1)) +
      String.fromCharCode(data.getUint8(2)) +
      String.fromCharCode(data.getUint8(3));
    if (magic !== 'NNUE') throw new Error('Invalid NNUE file header');
    offset += 4;

    const version = data.getInt32(offset, true);
    offset += 4;
    console.log(`Loading NNUE v${version}`);

    const readInt32 = () => {
      const v = data.getInt32(offset, true);
      offset += 4;
      return v;
    };

    const readFloat32Array = (len: number) => {
      const arr = new Float32Array(buffer, offset, len);
      offset += len * 4;
      return arr;
    };

    const fc1Len = readInt32();
    const inferredInputSize = fc1Len / 256;
    if (!Number.isInteger(inferredInputSize)) throw new Error(`Bad fc1_len=${fc1Len}`);
    this.inputSize = inferredInputSize;

    const fc1_weight: Float32Array[] = [];
    for (let i = 0; i < 256; i++) {
      fc1_weight.push(new Float32Array(buffer, offset + i * this.inputSize * 4, this.inputSize));
    }
    offset += fc1Len * 4;

    const fc1BiasLen = readInt32();
    const fc1_bias = readFloat32Array(fc1BiasLen);

    const fc2WeightLen = readInt32();
    const fc2_weight = readFloat32Array(fc2WeightLen);

    const fc2_bias = data.getFloat32(offset, true);
    offset += 4;

    this.weights = { fc1_weight, fc1_bias, fc2_weight, fc2_bias };
    console.log(`NNUE loaded, inputSize=${this.inputSize}`);
  }

  public setPosition(state: GameState, ply: number, history: HistoryState): void {
    if (!this.weights) {
      this.accumulator.fill(0);
      return;
    }

    const features = new Float32Array(this.inputSize);

    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const p = state.board[r][c];
        if (!p) continue;
        const col = pieceColor(p);
        const pt = pieceType(p);
        const idx = this.pieceMap[`${col}${pt}`] * 64 + (r * 8 + c);
        if (idx < this.inputSize) features[idx] = 1.0;
      }
    }

    if (this.inputSize > 768) {
      if (768 < this.inputSize) features[768] = isInCheck(state) ? 1 : 0;
      if (769 < this.inputSize) features[769] = Math.min(ply / 80, 1.0);
      if (770 < this.inputSize) features[770] = state.turn === 'w' ? 1 : 0;
      if (773 < this.inputSize) features[773] = Math.min(getAllLegalMoves(state).length / 30.0, 1.0);

      const key = boardKey(state);
      if (774 < this.inputSize) features[774] = Math.min(history.fenKeys.filter(k => k === key).length / 3, 1.0);
      if (775 < this.inputSize) features[775] = Math.min(history.noProgressPly / 20, 1.0);
    }

    const { fc1_weight, fc1_bias } = this.weights;
    for (let i = 0; i < 256; i++) {
      let sum = fc1_bias[i];
      const row = fc1_weight[i];
      for (let j = 0; j < this.inputSize; j++) {
        sum += row[j] * features[j];
      }
      if (sum < 0) sum = 0;
      if (sum > 1) sum = 1;
      this.accumulator[i] = sum;
    }
  }

  public evaluate(): number {
    if (!this.weights) return 0;
    let out = this.weights.fc2_bias;
    for (let i = 0; i < 256; i++) {
      out += this.accumulator[i] * this.weights.fc2_weight[i];
    }
    return out * 100;
  }
}

const evaluator = new NNUEEvaluator();

// ──────────────────────────────────────────────────────────────────────────────
// Zobrist / TT
// ──────────────────────────────────────────────────────────────────────────────

const ZOBRIST_PIECES = new Uint32Array(12 * 64);
const ZOBRIST_SIDE = BigInt('0x123456789ABCDEF0');

function initZobrist() {
  for (let i = 0; i < ZOBRIST_PIECES.length; i++) {
    ZOBRIST_PIECES[i] = Math.floor(Math.random() * 0x100000000);
  }
}
initZobrist();

function computeZobrist(state: GameState): bigint {
  let hash = 0n;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = state.board[r][c];
      if (!p) continue;
      const pt = pieceType(p);
      const pc = pieceColor(p);
      const pieceIdx = (pc === 'w' ? 0 : 6) + 'PNBRQK'.indexOf(pt);
      const sq = r * 8 + c;
      hash ^= BigInt(ZOBRIST_PIECES[pieceIdx * 64 + sq]);
    }
  }
  if (state.turn === 'b') hash ^= ZOBRIST_SIDE;
  return hash;
}

interface TTEntry {
  depth: number;
  score: number;
  flag: 0 | 1 | 2;
  move?: Move;
}

const TT = new Map<bigint, TTEntry>();
const TT_SIZE = 1 << 20;

function getTTMove(hash: bigint): Move | undefined {
  return TT.get(hash)?.move;
}

function storeTT(hash: bigint, depth: number, score: number, flag: 0 | 1 | 2, move?: Move) {
  const existing = TT.get(hash);
  if (!existing || depth >= existing.depth) {
    TT.set(hash, { depth, score, flag, move });
    if (TT.size > TT_SIZE) {
      const firstKey = TT.keys().next().value;
      TT.delete(firstKey);
    }
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tactical helpers
// ──────────────────────────────────────────────────────────────────────────────

function getAttackingMoves(state: GameState, target: { r: number; c: number }): Move[] {
  return getAllLegalMoves(state).filter(m => m.to.r === target.r && m.to.c === target.c);
}

function countDefenders(state: GameState, target: { r: number; c: number }, defenderTurn: 'w' | 'b'): number {
  const defenderState: GameState = { ...state, turn: defenderTurn };
  return getAttackingMoves(defenderState, target).length;
}

function movedPieceAt(nextState: GameState, move: Move): string | null {
  return nextState.board[move.to.r]?.[move.to.c] ?? null;
}

function cheapestAttackerValue(state: GameState, target: { r: number; c: number }): number | null {
  const attacks = getAttackingMoves(state, target);
  if (!attacks.length) return null;

  let best = Infinity;
  for (const a of attacks) {
    const v = PIECE_VALUE[pieceType(a.piece)] ?? 10000;
    if (v < best) best = v;
  }
  return best === Infinity ? null : best;
}

function hangingPenalty(prevState: GameState, nextState: GameState, move: Move): number {
  const moved = movedPieceAt(nextState, move);
  if (!moved) return 0;

  const movedType = pieceType(moved);
  if (movedType === 'K') return 0;

  const movedVal = PIECE_VALUE[movedType] ?? 0;
  const attackers = getAttackingMoves(nextState, move.to);
  if (!attackers.length) return 0;

  const defenders = countDefenders(nextState, move.to, prevState.turn);
  const cheapest = cheapestAttackerValue(nextState, move.to) ?? movedVal;

  let penalty = 0;

  if (defenders === 0) penalty += Math.floor(movedVal * 0.85);
  if (cheapest < movedVal) penalty += Math.floor((movedVal - cheapest) * 0.7);
  if (attackers.length > defenders) penalty += Math.min((attackers.length - defenders) * 60, 180);
  if (movedType === 'Q') penalty = Math.floor(penalty * 1.5);

  return penalty;
}

function suicidalCheckPenalty(prevState: GameState, nextState: GameState, move: Move): number {
  if (!isInCheck(nextState)) return 0;

  const moved = movedPieceAt(nextState, move);
  if (!moved) return 0;

  const movedType = pieceType(moved);
  if (movedType === 'K') return 0;

  const movedVal = PIECE_VALUE[movedType] ?? 0;
  const cheapest = cheapestAttackerValue(nextState, move.to);
  if (cheapest == null) return 0;

  if (cheapest < movedVal) {
    return Math.floor((movedVal - cheapest) * 0.9);
  }
  return 0;
}

function losingCapturePenalty(prevState: GameState, nextState: GameState, move: Move): number {
  if (!move.captured) return 0;
  const attacker = PIECE_VALUE[pieceType(move.piece)] ?? 0;
  const victim = PIECE_VALUE[pieceType(move.captured)] ?? 0;

  const cheapestRecapture = cheapestAttackerValue(nextState, move.to);
  if (cheapestRecapture == null) return 0;

  if (victim < attacker && cheapestRecapture <= attacker) {
    return Math.floor((attacker - victim) * 0.8);
  }
  return 0;
}

// ──────────────────────────────────────────────────────────────────────────────
// Positional penalties/bonuses
// ──────────────────────────────────────────────────────────────────────────────

function antiShufflePenalty(state: GameState, history: HistoryState, lastMove?: Move): number {
  let penalty = 0;

  const key = boardKey(state);
  const reps = history.fenKeys.filter(k => k === key).length;
  if (reps >= 1) penalty += 35;
  if (reps >= 2) penalty += 80;
  if (reps >= 3) penalty += 200;

  const prev = history.moves[history.moves.length - 1];
  const prev2 = history.moves[history.moves.length - 2];

  if (lastMove) {
    if (isReverseMove(lastMove, prev)) penalty += 120;
    if (isRookShuffle(lastMove, prev)) penalty += 180;
    if (prev2 && isReverseMove(lastMove, prev2)) penalty += 60;
  }

  penalty += Math.min(history.noProgressPly * 4, 120);
  return penalty;
}

function progressBonus(move?: Move, state?: GameState): number {
  if (!move || !state) return 0;
  let bonus = 0;
  if (move.captured) bonus += 25;
  if (move.promotion) bonus += 250;
  if (isInCheck(state)) bonus += 30;
  return bonus;
}

// ──────────────────────────────────────────────────────────────────────────────
// Evaluation
// ──────────────────────────────────────────────────────────────────────────────

function evalWithContext(
  state: GameState,
  ply: number,
  history: HistoryState,
  lastMove?: Move,
  prevState?: GameState,
): number {
  const result = getGameResult(state);

  if (result === 'checkmate') return -99999 + ply;
  if (result === 'stalemate' || result === 'draw-50') return 0;

  evaluator.setPosition(state, ply, history);
  let score = evaluator.evaluate();

  score -= antiShufflePenalty(state, history, lastMove);

  if (lastMove && prevState) {
    score -= hangingPenalty(prevState, state, lastMove);
    score -= suicidalCheckPenalty(prevState, state, lastMove);
    score -= losingCapturePenalty(prevState, state, lastMove);
  }

  score += progressBonus(lastMove, state);
  return score;
}

export function evaluate(state: GameState): number {
  const history: HistoryState = {
    fenKeys: [boardKey(state)],
    moves: [],
    noProgressPly: 0,
  };
  return evalWithContext(state, 0, history);
}

// ──────────────────────────────────────────────────────────────────────────────
// Move ordering
// ──────────────────────────────────────────────────────────────────────────────

function scoreMoveForOrdering(
  state: GameState,
  move: Move,
  ply: number,
  ttMove?: Move,
  history?: HistoryState,
): number {
  if (ttMove && areMovesEqual(move, ttMove)) return 10_000_000;

  let score = 0;
  const next = applyMove(state, move);

  if (move.captured) {
    const attacker = PIECE_VALUE[pieceType(move.piece)] ?? 0;
    const victim = PIECE_VALUE[pieceType(move.captured)] ?? 0;
    score += 100_000 + victim * 10 - attacker;
  }

  if (move.promotion) {
    score += 80_000 + (PIECE_VALUE[move.promotion] ?? 0);
  }

  if (isInCheck(next)) {
    score += 50_000;
  }

  const k1 = killerMoves[ply]?.[0];
  const k2 = killerMoves[ply]?.[1];
  if (k1 && areMovesEqual(move, k1)) score += 40_000;
  else if (k2 && areMovesEqual(move, k2)) score += 30_000;

  score += historyHeuristic[moveKey(move)] ?? 0;

  score -= hangingPenalty(state, next, move);
  score -= suicidalCheckPenalty(state, next, move);
  score -= losingCapturePenalty(state, next, move);

  if (history) {
    const nextHistory = pushHistory(history, next, move);
    score -= antiShufflePenalty(next, nextHistory, move);
  }

  return score;
}

function orderMoves(
  state: GameState,
  moves: Move[],
  ply: number,
  ttMove?: Move,
  history?: HistoryState,
): Move[] {
  return [...moves].sort(
    (a, b) =>
      scoreMoveForOrdering(state, b, ply, ttMove, history) -
      scoreMoveForOrdering(state, a, ply, ttMove, history)
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Quiescence
// ──────────────────────────────────────────────────────────────────────────────

let nodesSearched = 0;

function isInterestingQMove(state: GameState, move: Move): boolean {
  const next = applyMove(state, move);
  if (move.captured) {
    const penalty = losingCapturePenalty(state, next, move);
    return penalty < 180;
  }
  if (isInCheck(next)) return true;
  if (move.promotion) return true;
  return false;
}

function quiescence(
  state: GameState,
  alpha: number,
  beta: number,
  ply: number,
  history: HistoryState,
  lastMove?: Move,
  prevState?: GameState,
): number {
  const standPat = evalWithContext(state, ply, history, lastMove, prevState);
  if (standPat >= beta) return beta;
  if (standPat > alpha) alpha = standPat;

  const forcingMoves = getAllLegalMoves(state).filter(m => isInterestingQMove(state, m));
  if (forcingMoves.length === 0) return alpha;

  const ordered = orderMoves(state, forcingMoves, ply, undefined, history);
  for (const move of ordered) {
    const next = applyMove(state, move);
    const nextHistory = pushHistory(history, next, move);

    const score = -quiescence(next, -beta, -alpha, ply + 1, nextHistory, move, state);
    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }

  return alpha;
}

// ──────────────────────────────────────────────────────────────────────────────
// Negamax
// ──────────────────────────────────────────────────────────────────────────────

function negamax(
  state: GameState,
  depth: number,
  alpha: number,
  beta: number,
  ply: number,
  history: HistoryState,
  lastMove?: Move,
  prevState?: GameState,
): number {
  nodesSearched++;

  const hash = computeZobrist(state);
  const ttEntry = TT.get(hash);

  if (ttEntry && ttEntry.depth >= depth) {
    if (ttEntry.flag === 0) return ttEntry.score;
    if (ttEntry.flag === 1 && ttEntry.score >= beta) return ttEntry.score;
    if (ttEntry.flag === 2 && ttEntry.score <= alpha) return ttEntry.score;
  }

  const result = getGameResult(state);
  if (result !== 'ongoing') {
    if (result === 'checkmate') return -99999 + ply;
    return 0;
  }

  if (depth === 0) {
    return quiescence(state, alpha, beta, ply, history, lastMove, prevState);
  }

  const alphaOrig = alpha;
  const ttMove = ttEntry?.move;
  const legalMoves = getAllLegalMoves(state);
  const moves = orderMoves(state, legalMoves, ply, ttMove, history);

  let bestScore = -Infinity;
  let bestMove: Move | undefined;

  for (const move of moves) {
    const next = applyMove(state, move);
    const nextHistory = pushHistory(history, next, move);

    const score = -negamax(
      next,
      depth - 1,
      -beta,
      -alpha,
      ply + 1,
      nextHistory,
      move,
      state
    );

    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }

    if (score > alpha) alpha = score;

    if (alpha >= beta) {
      if (!move.captured) {
        recordKiller(ply, move);
        recordHistory(move, depth);
      }
      break;
    }
  }

  let flag: 0 | 1 | 2 = 0;
  if (bestScore <= alphaOrig) flag = 2;
  else if (bestScore >= beta) flag = 1;
  else flag = 0;

  storeTT(hash, depth, bestScore, flag, bestMove);
  return bestScore;
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

export interface AIResult {
  move: Move;
  score: number;
  depth: number;
  nodes: number;
}

function buildRootHistory(state: GameState): HistoryState {
  return {
    fenKeys: [boardKey(state)],
    moves: [],
    noProgressPly: 0,
  };
}

export async function getBestMove(
  state: GameState,
  timeLimitMs = 1500,
): Promise<AIResult> {
  return new Promise((resolve) => {
    setTimeout(() => {
      const start = Date.now();
      nodesSearched = 0;
      TT.clear();

      const moves = getAllLegalMoves(state);
      if (moves.length === 0) throw new Error('No legal moves');
      if (moves.length === 1) {
        resolve({ move: moves[0], score: 0, depth: 0, nodes: 1 });
        return;
      }

      const rootHistory = buildRootHistory(state);

      let bestMove = moves[0];
      let bestScore = -Infinity;
      let completedDepth = 0;

      for (let depth = 1; depth <= 10; depth++) {
        if (Date.now() - start > timeLimitMs) break;

        let depthBestMove = moves[0];
        let depthBestScore = -Infinity;
        let alpha = -Infinity;
        let beta = Infinity;

        const hash = computeZobrist(state);
        const ttMove = getTTMove(hash);
        const ordered = orderMoves(state, moves, 0, ttMove, rootHistory);

        for (const move of ordered) {
          if (Date.now() - start > timeLimitMs) break;

          const next = applyMove(state, move);
          const nextHistory = pushHistory(rootHistory, next, move);

          const score = -negamax(
            next,
            depth - 1,
            -beta,
            -alpha,
            1,
            nextHistory,
            move,
            state
          );

          if (score > depthBestScore) {
            depthBestScore = score;
            depthBestMove = move;
          }
          if (score > alpha) alpha = score;
        }

        bestMove = depthBestMove;
        bestScore = depthBestScore;
        completedDepth = depth;
      }

      resolve({
        move: bestMove,
        score: bestScore,
        depth: completedDepth,
        nodes: nodesSearched,
      });
    }, 10);
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────────────────────────────────────

let initialized = false;

export async function initAI(modelUrl: string = '/best_network.nnue'): Promise<void> {
  if (initialized) return;
  await evaluator.loadWeights(modelUrl);
  initialized = true;
  console.log('AI initialized with NNUE + stronger search heuristics');
}
