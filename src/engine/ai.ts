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
// History
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

function sqEq(a: { r: number; c: number }, b: { r: number; c: number }): boolean {
  return a.r === b.r && a.c === b.c;
}

function sameMove(a?: Move, b?: Move): boolean {
  if (!a || !b) return false;
  return (
    sqEq(a.from, b.from) &&
    sqEq(a.to, b.to) &&
    a.promotion === b.promotion
  );
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
    sqEq(current.from, prev.to) &&
    sqEq(current.to, prev.from) &&
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
  const capture = !!move.captured;
  const promotion = !!move.promotion;
  if (pawnMove || capture || promotion) return 0;
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
// NNUE evaluator
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
    if (!Number.isInteger(inferredInputSize)) {
      throw new Error(`fc1_len ${fc1Len} not divisible by 256`);
    }
    this.inputSize = inferredInputSize;

    const fc1_weight: Float32Array[] = [];
    for (let i = 0; i < 256; i++) {
      fc1_weight.push(
        new Float32Array(buffer, offset + i * this.inputSize * 4, this.inputSize)
      );
    }
    offset += fc1Len * 4;

    const fc1BiasLen = readInt32();
    const fc1_bias = readFloat32Array(fc1BiasLen);

    const fc2WeightLen = readInt32();
    const fc2_weight = readFloat32Array(fc2WeightLen);

    const fc2_bias = data.getFloat32(offset, true);
    offset += 4;

    this.weights = { fc1_weight, fc1_bias, fc2_weight, fc2_bias };
    console.log(`NNUE weights loaded, inputSize=${this.inputSize}`);
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
        const key = `${col}${pt}`;
        const pieceIdx = this.pieceMap[key];
        if (pieceIdx === undefined) continue;
        const sq = r * 8 + c;
        const idx = pieceIdx * 64 + sq;
        if (idx < this.inputSize) features[idx] = 1.0;
      }
    }

    if (this.inputSize > 768) {
      if (768 < this.inputSize) features[768] = isInCheck(state) ? 1 : 0;
      if (769 < this.inputSize) features[769] = Math.min(ply / 80, 1.0);
      if (770 < this.inputSize) features[770] = state.turn === 'w' ? 1 : 0;

      let wKr = -1, wKc = -1, bKr = -1, bKc = -1;
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const p = state.board[r][c];
          if (!p || pieceType(p) !== 'K') continue;
          if (pieceColor(p) === 'w') {
            wKr = r; wKc = c;
          } else {
            bKr = r; bKc = c;
          }
        }
      }

      const centerR = 3.5;
      const centerC = 3.5;
      if (771 < this.inputSize && wKr !== -1) {
        const dr = wKr - centerR;
        const dc = wKc - centerC;
        features[771] = Math.min(Math.sqrt(dr * dr + dc * dc) / 4.0, 1.0);
      }
      if (772 < this.inputSize && bKr !== -1) {
        const dr = bKr - centerR;
        const dc = bKc - centerC;
        features[772] = Math.min(Math.sqrt(dr * dr + dc * dc) / 4.0, 1.0);
      }
      if (773 < this.inputSize) {
        features[773] = Math.min(getAllLegalMoves(state).length / 30.0, 1.0);
      }
      if (774 < this.inputSize) {
        const key = boardKey(state);
        const reps = history.fenKeys.filter((k) => k === key).length;
        features[774] = Math.min(reps / 3, 1.0);
      }
      if (775 < this.inputSize) {
        features[775] = Math.min(history.noProgressPly / 20, 1.0);
      }
    }

    const { fc1_weight, fc1_bias } = this.weights;
    for (let i = 0; i < 256; i++) {
      let sum = fc1_bias[i];
      const row = fc1_weight[i];
      for (let j = 0; j < this.inputSize; j++) {
        sum += row[j] * features[j];
      }
      if (sum < 0) sum = 0;
      else if (sum > 1) sum = 1;
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

const TT_SIZE = 1 << 20;
const TT = new Map<bigint, TTEntry>();

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
// Attack / hanging piece helpers
// ──────────────────────────────────────────────────────────────────────────────

function findMovedPieceOnBoard(state: GameState, move: Move): string | null {
  const piece = state.board[move.to.r]?.[move.to.c] ?? null;
  if (!piece) return null;
  return piece;
}

function getAttackingMoves(state: GameState, target: { r: number; c: number }): Move[] {
  const legal = getAllLegalMoves(state);
  return legal.filter((m) => m.to.r === target.r && m.to.c === target.c);
}

function getCheapestAttackerValue(state: GameState, target: { r: number; c: number }): number | null {
  const attacks = getAttackingMoves(state, target);
  if (attacks.length === 0) return null;

  let best = Infinity;
  for (const m of attacks) {
    const p = m.piece;
    if (!p) continue;
    const val = PIECE_VALUE[pieceType(p)] ?? 1000;
    if (val < best) best = val;
  }
  return best === Infinity ? null : best;
}

function squareDefenderCount(state: GameState, target: { r: number; c: number }): number {
  return getAttackingMoves(state, target).length;
}

function hangingPenalty(
  prevState: GameState,
  nextState: GameState,
  move: Move,
): number {
  const movedPiece = findMovedPieceOnBoard(nextState, move);
  if (!movedPiece) return 0;

  const movedType = pieceType(movedPiece);
  const movedValue = PIECE_VALUE[movedType] ?? 0;
  if (movedType === 'K') return 0;

  // Opponent to move in nextState
  const oppAttacks = getAttackingMoves(nextState, move.to);
  if (oppAttacks.length === 0) return 0;

  const cheapestAttacker = getCheapestAttackerValue(nextState, move.to) ?? 0;

  // Count our defenders by flipping turn back temporarily
  const sameSideState: GameState = {
    ...nextState,
    turn: prevState.turn,
  };
  const defenders = squareDefenderCount(sameSideState, move.to);
  const attackers = oppAttacks.length;

  let penalty = 0;

  // Pure hanging piece
  if (defenders === 0) {
    penalty += Math.floor(movedValue * 0.85);
  }

  // Bad trade warning
  if (cheapestAttacker < movedValue) {
    penalty += Math.floor((movedValue - cheapestAttacker) * 0.7);
  }

  // Outnumbered on destination square
  if (attackers > defenders) {
    penalty += Math.min((attackers - defenders) * 60, 180);
  }

  // Extra queen-safety punishment
  if (movedType === 'Q') {
    penalty = Math.floor(penalty * 1.5);
  }

  return penalty;
}

function suicidalCheckPenalty(prevState: GameState, nextState: GameState, move: Move): number {
  const givesCheck = isInCheck(nextState);
  if (!givesCheck) return 0;

  const movedPiece = findMovedPieceOnBoard(nextState, move);
  if (!movedPiece) return 0;

  const movedType = pieceType(movedPiece);
  const movedValue = PIECE_VALUE[movedType] ?? 0;

  const oppAttacks = getAttackingMoves(nextState, move.to);
  if (oppAttacks.length === 0) return 0;

  const cheapestAttacker = getCheapestAttackerValue(nextState, move.to) ?? movedValue;
  if (cheapestAttacker < movedValue) {
    return Math.floor((movedValue - cheapestAttacker) * 0.9);
  }
  return 0;
}

// ──────────────────────────────────────────────────────────────────────────────
// Anti-shuffle / progress
// ──────────────────────────────────────────────────────────────────────────────

function antiShufflePenalty(state: GameState, history: HistoryState, lastMove?: Move): number {
  let penalty = 0;

  const key = boardKey(state);
  const reps = history.fenKeys.filter((k) => k === key).length;

  if (reps >= 1) penalty += 35;
  if (reps >= 2) penalty += 80;
  if (reps >= 3) penalty += 200;

  const prev = history.moves[history.moves.length - 1];
  const prev2 = history.moves[history.moves.length - 2];
  const prev3 = history.moves[history.moves.length - 3];

  if (lastMove) {
    if (isReverseMove(lastMove, prev)) penalty += 120;
    if (isRookShuffle(lastMove, prev)) penalty += 180;
    if (prev2 && isReverseMove(lastMove, prev2)) penalty += 60;

    if (
      prev &&
      lastMove.piece &&
      prev.piece &&
      lastMove.piece === prev.piece &&
      !lastMove.captured &&
      !lastMove.promotion
    ) {
      penalty += 20;
    }

    if (
      prev3 &&
      lastMove.piece &&
      prev3.piece &&
      lastMove.piece === prev3.piece &&
      !lastMove.captured
    ) {
      penalty += 20;
    }
  }

  penalty += Math.min(history.noProgressPly * 4, 120);
  return penalty;
}

function progressBonus(move?: Move, nextState?: GameState): number {
  if (!move || !nextState) return 0;
  let bonus = 0;
  if (move.captured) bonus += 25;
  if (move.promotion) bonus += 250;
  if (isInCheck(nextState)) bonus += 30;
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

function areMovesEqual(a?: Move, b?: Move): boolean {
  if (!a || !b) return false;
  return (
    a.from.r === b.from.r &&
    a.from.c === b.from.c &&
    a.to.r === b.to.r &&
    a.to.c === b.to.c &&
    a.promotion === b.promotion
  );
}

function scoreMoveForOrdering(
  state: GameState,
  move: Move,
  ttMove?: Move,
  history?: HistoryState,
): number {
  if (ttMove && areMovesEqual(move, ttMove)) return 1_000_000;

  let score = 0;
  const next = applyMove(state, move);

  if (isInCheck(next)) score += 50_000;

  if (move.captured) {
    const attacker = PIECE_VALUE[pieceType(move.piece)] ?? 0;
    const victim = PIECE_VALUE[pieceType(move.captured)] ?? 0;
    score += 10_000 + (victim * 10 - attacker);
  }

  if (move.promotion) {
    score += 20_000 + (PIECE_VALUE[move.promotion] ?? 0);
  }

  if (pieceType(move.piece) === 'R') {
    score -= 25;
  }

  // Immediate anti-blunder ordering
  score -= hangingPenalty(state, next, move);
  score -= suicidalCheckPenalty(state, next, move);

  if (history) {
    const nextHistory = pushHistory(history, next, move);
    score -= antiShufflePenalty(next, nextHistory, move);
  }

  return score;
}

function orderMoves(
  state: GameState,
  moves: Move[],
  ttMove?: Move,
  history?: HistoryState,
): Move[] {
  return [...moves].sort(
    (a, b) =>
      scoreMoveForOrdering(state, b, ttMove, history) -
      scoreMoveForOrdering(state, a, ttMove, history),
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Search
// ──────────────────────────────────────────────────────────────────────────────

let nodesSearched = 0;

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

  const forcingMoves = getAllLegalMoves(state).filter((m) => {
    return !!m.captured || isInCheck(applyMove(state, m));
  });

  if (forcingMoves.length === 0) return alpha;

  const ordered = orderMoves(state, forcingMoves, undefined, history);
  for (const move of ordered) {
    const next = applyMove(state, move);
    const nextHistory = pushHistory(history, next, move);
    const score = -quiescence(next, -beta, -alpha, ply + 1, nextHistory, move, state);

    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }

  return alpha;
}

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

  const ttMove = ttEntry?.move;
  const moves = orderMoves(state, getAllLegalMoves(state), ttMove, history);

  let bestScore = -Infinity;
  let bestMove: Move | undefined = undefined;
  const alphaOrig = alpha;

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
      state,
    );

    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
    if (score > alpha) alpha = score;
    if (alpha >= beta) break;
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

      for (let depth = 1; depth <= 8; depth++) {
        if (Date.now() - start > timeLimitMs) break;

        let depthBestMove = moves[0];
        let depthBestScore = -Infinity;
        let alpha = -Infinity;
        const beta = Infinity;

        const hash = computeZobrist(state);
        const ttMove = getTTMove(hash);
        const ordered = orderMoves(state, moves, ttMove, rootHistory);

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
            state,
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
  console.log('AI initialized with trained NNUE + anti-shuffle + anti-blunder penalties');
}
